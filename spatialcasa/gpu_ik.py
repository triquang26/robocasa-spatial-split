import os

import mujoco
import numpy as np
import torch

from spatialcasa.ik import IK


def quat_mat(q):
    m = np.zeros(9)
    mujoco.mju_quat2Mat(m, q)
    return m.reshape(3, 3)


def skew(a):
    return np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])


def hinge(axis, th):
    a = axis / np.linalg.norm(axis)
    K = skew(a)
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * K @ K


BUCKETS = (32, 256, 2048, 16384)
CHUNK = 10


def free_gpu():
    want = os.environ.get("SPATIALCASA_IK_DEVICE")
    if want:
        return want
    if not torch.cuda.is_available():
        return "cpu"
    free = [torch.cuda.mem_get_info(i)[0] for i in range(torch.cuda.device_count())]
    return f"cuda:{int(np.argmax(free))}"


def spd_solve(A, b):
    n = A.shape[-1]
    L = [[None] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1):
            acc = A[:, i, j] - sum(L[i][k] * L[j][k] for k in range(j)) if j else A[:, i, j]
            L[i][j] = torch.sqrt(acc) if i == j else acc / L[j][j]
    y = [None] * n
    for i in range(n):
        y[i] = (b[:, i] - sum(L[i][k] * y[k] for k in range(i))) / L[i][i] if i else b[:, i] / L[i][i]
    x = [None] * n
    for i in reversed(range(n)):
        x[i] = (y[i] - sum(L[k][i] * x[k] for k in range(i + 1, n))) / L[i][i] if i < n - 1 else y[i] / L[i][i]
    return torch.stack(x, 1)


class BatchIK:
    def __init__(self, world, side, use_waist=True, device=None):
        self.device = device or free_gpu()
        t = lambda x: torch.tensor(np.asarray(x, float), dtype=torch.float64, device=self.device)
        ref = IK(world, side, use_waist)
        m = ref.m
        self.n = len(ref.qadr)
        self.lo, self.hi, self.rest = t(ref.lo), t(ref.hi), t(ref.rest)
        index = {a: k for k, a in enumerate(ref.qadr)}
        qpos = world.sim.data.qpos
        chain = []
        body = m.site_bodyid[ref.site]
        while body != 0:
            chain.append(body)
            body = m.body_parentid[body]
        p, R = np.zeros(3), np.eye(3)
        self.segments = []
        for b in chain[::-1]:
            p, R = p + R @ m.body_pos[b], R @ quat_mat(m.body_quat[b])
            for j in range(m.body_jntadr[b], m.body_jntadr[b] + m.body_jntnum[b]):
                if m.jnt_type[j] != mujoco.mjtJoint.mjJNT_HINGE:
                    raise ValueError("only hinge joints are supported")
                k = index.get(m.jnt_qposadr[j])
                if k is None:
                    Rj = hinge(m.jnt_axis[j], qpos[m.jnt_qposadr[j]])
                    anchor = p + R @ m.jnt_pos[j]
                    R = R @ Rj
                    p = anchor - R @ m.jnt_pos[j]
                    continue
                K = skew(m.jnt_axis[j] / np.linalg.norm(m.jnt_axis[j]))
                self.segments.append((t(p), t(R), t(m.jnt_pos[j]), t(K), t(K @ K), t(m.jnt_axis[j] / np.linalg.norm(m.jnt_axis[j])), k))
                p, R = np.zeros(3), np.eye(3)
        p, R = p + R @ m.site_pos[ref.site], R @ quat_mat(m.site_quat[ref.site])
        self.tail = (t(p), t(R))
        self.eye3 = t(np.eye(3))
        self.eye6 = t(np.eye(6))
        mode = os.environ.get("SPATIALCASA_IK_COMPILE", "reduce-overhead")
        self._step = self.step if self.device == "cpu" or mode == "none" else \
            torch.compile(self.step, mode=mode, dynamic=False)

    def fk(self, q):
        B = q.shape[0]
        p = torch.zeros(B, 3, dtype=q.dtype, device=q.device)
        R = self.eye3.expand(B, 3, 3)
        axes, anchors = [None] * self.n, [None] * self.n
        for cp, cR, jpos, K, K2, axis, k in self.segments:
            p = p + R @ cp
            R = R @ cR
            anchor = p + R @ jpos
            axes[k], anchors[k] = R @ axis, anchor
            th = q[:, k, None, None]
            R = R @ (self.eye3 + torch.sin(th) * K + (1 - torch.cos(th)) * K2)
            p = anchor - R @ jpos
        ps = p + R @ self.tail[0]
        Rs = R @ self.tail[1]
        a = torch.stack(axes, -1)
        J = torch.cat([torch.cross(a, ps[..., None] - torch.stack(anchors, -1), dim=1), a], 1)
        return ps, Rs, J

    def step(self, q, pos, rot, free, W, damping, w_rest, tol0, tol1):
        p, R, J = self.fk(q)
        E = rot @ R.transpose(1, 2)
        er = 0.5 * torch.stack([E[:, 2, 1] - E[:, 1, 2], E[:, 0, 2] - E[:, 2, 0], E[:, 1, 0] - E[:, 0, 1]], -1)
        ep = pos - p
        done = (torch.linalg.norm(ep, dim=-1) < tol0) & (torch.linalg.norm(er, dim=-1) < tol1)
        Jw = J * W[None, :, None] * free
        Jt = Jw.transpose(1, 2)
        A = Jw @ Jt + damping ** 2 * self.eye6
        r = w_rest * (self.rest - q)
        rhs = torch.cat([ep, er], -1) * W - (Jw @ r[..., None])[..., 0]
        dq = r + (Jt @ spd_solve(A, rhs)[..., None])[..., 0]
        step = free * dq.clamp(-0.2, 0.2) * (~done)[:, None]
        return torch.minimum(torch.maximum(q + step, self.lo), self.hi), done

    def solve_many(self, pos, rot, waist=None):
        qs, oks = [], []
        for a in range(0, len(pos), BUCKETS[-1]):
            b = slice(a, a + BUCKETS[-1])
            q, ok = self.solve(pos[b], rot[b], waist=None if waist is None else waist[b])
            qs.append(q)
            oks.append(ok)
        return torch.cat(qs), torch.cat(oks)

    def solve(self, pos, rot, q0=None, waist=None, iters=300, damping=0.05, w_rot=0.3, w_rest=0.02,
              tol=(0.002, 0.03)):
        f64 = dict(dtype=torch.float64, device=self.device)
        pos, rot = torch.as_tensor(pos, **f64), torch.as_tensor(rot, **f64)
        B = pos.shape[0]
        n = next(b for b in BUCKETS if b >= B) if B <= BUCKETS[-1] else B
        q = (self.rest.expand(B, self.n) if q0 is None else torch.as_tensor(q0, **f64)).clone()
        free = torch.ones(self.n, **f64)
        if waist is not None:
            q[:, 7:] = torch.as_tensor(waist, **f64)
            free[7:] = 0
        pad = n - B
        if pad:
            pos = torch.cat([pos, pos[:1].expand(pad, 3)])
            rot = torch.cat([rot, rot[:1].expand(pad, 3, 3)])
            q = torch.cat([q, q[:1].expand(pad, self.n)])
        W = torch.tensor([1, 1, 1, w_rot, w_rot, w_rot], **f64)
        for it in range(0, iters, CHUNK):
            for _ in range(CHUNK):
                q, done = self._step(q, pos, rot, free, W, damping, w_rest, tol[0], tol[1])
                q = q.clone()
            if bool(done.all()):
                break
        q = q[:B]
        p, R, _ = self.fk(q)
        E = rot[:B] @ R.transpose(1, 2)
        ang = torch.arccos(((E.diagonal(dim1=1, dim2=2).sum(-1) - 1) / 2).clamp(-1, 1))
        ok = (torch.linalg.norm(pos[:B] - p, dim=-1) < 3 * tol[0]) & (ang < 3 * tol[1])
        return q, ok
