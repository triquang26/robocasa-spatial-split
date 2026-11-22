import copy
import json
import os

import mujoco
import numpy as np

from spatialcasa import relations as REL
from spatialcasa.ik import IK, arm_joints

UP = np.array([0.0, 0.0, 1.0])
HZ = 20
GRASP_HEIGHT = 0.08
GRASP_OFFSET = (0.02, -0.08, 0.05)
PRE_BACK = 0.10
PRE_UP = 0.03
CARRY_CLEAR = 0.06
RELEASE_LIFT = 0.012
SPEED = 0.25
SLOW = 0.08
SETTLE = 8
GRIP_CLOSED = 1.0
SPIN = np.radians([0, 20, -20, 40, -40, 60, -60, 80, -80])
SIDES = {"right": (slice(0, 7), slice(17, 23), 1.0), "left": (slice(7, 14), slice(23, 29), -1.0)}
TORSO = slice(14, 17)
MAX_DISTURB = 0.02
MAX_TILT = 15.0
WAIST_GRID = np.array([(y, p, 0.0) for y in np.arange(-0.9, 0.91, 0.1) for p in np.arange(0.0, 0.91, 0.1)])
WAIST_COST = np.abs(WAIST_GRID).sum(1)
PAIRS = 24
TRIES = 4
CHECK_STRIDE = 2
GRASPS = {
    "side": {"height": GRASP_HEIGHT, "top": 0.0, "inset": None, "pitch": 0.0, "roll": 0.0,
             "offset": GRASP_OFFSET, "open": 0.0, "close": GRIP_CLOSED},
    "rim": {"height": 0.0, "top": 1.0, "inset": 0.0, "pitch": -1.5708, "roll": 1.5708,
            "offset": GRASP_OFFSET, "open": -1.0, "close": GRIP_CLOSED},
}
_GPU = {}
_IK_DEVICE = [None]


def batch_ik(world, side):
    from spatialcasa.gpu_ik import BatchIK
    key = (id(world), side)
    if key not in _GPU:
        _GPU[key] = BatchIK(world, side, device=_IK_DEVICE[0])
    return _GPU[key]


def ik_to_cpu(err):
    import torch
    print(f"batch IK falls back to cpu: {type(err).__name__}: {str(err).splitlines()[0]}", flush=True)
    _IK_DEVICE[0] = "cpu"
    _GPU.clear()
    try:
        torch.cuda.empty_cache()
    except Exception:
        pass


def grasp_profile(name=None, override=None):
    from spatialcasa import task as T
    c = T.spec()["candidate"]
    g = dict(GRASPS[name or c.get("grasp", "side")])
    g.update(c.get("grasp_params") or {})
    env = os.environ.get("SPATIALCASA_GRASP")
    if env:
        g.update(json.loads(env))
    g.update(override or {})
    return g


class PlanError(Exception):
    pass


def rot_z(th):
    c, s = np.cos(th), np.sin(th)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def rot_axis(axis, th):
    a = np.asarray(axis, float) / np.linalg.norm(axis)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * K @ K


def smooth(n):
    t = np.linspace(0, 1, n + 1)[1:]
    return t * t * (3 - 2 * t)


class Key:
    def __init__(self, center, spin, back=0.0, up=0.0):
        self.center = np.asarray(center, float)
        self.spin, self.back, self.up = float(spin), float(back), float(up)

    def mix(self, other, s):
        return Key(self.center + s * (other.center - self.center), self.spin + s * (other.spin - self.spin),
                   self.back + s * (other.back - self.back), self.up + s * (other.up - self.up))


def subtree(m, root):
    out = {root}
    for b in range(root + 1, m.nbody):
        if m.body_parentid[b] in out:
            out.add(b)
    return out


def collision_model(world):
    if getattr(world, "_col", None) is None:
        m = copy.deepcopy(world.sim.model._model)
        world._col = (m, mujoco.MjData(m))
    return world._col


class Planner:
    def __init__(self, world, side, obstacles=(), grasp=None, half=None):
        self.w = world
        self.side = side
        self.g = grasp or GRASPS["side"]
        self.half = np.zeros(3) if half is None else np.asarray(half, float)
        self.full = IK(world, side)
        self.arm = IK(world, side, use_waist=False)
        self.qadr = self.full.qadr
        yaw, sgn = world.frame.yaw, SIDES[side][2]
        a = np.array([-np.sin(yaw) * sgn, np.cos(yaw) * sgn, 0.0])
        x = -UP if side == "right" else UP
        R0 = np.column_stack([x, np.cross(a, x), a])
        if self.g.get("pitch"):
            R0 = R0 @ rot_axis([0, 1, 0], self.g["pitch"] * (1.0 if side == "left" else -1.0))
        if self.g.get("roll"):
            R0 = R0 @ rot_axis([0, 0, 1], self.g["roll"])
        self.R0 = R0
        self.a0 = a
        self.offset = np.array(self.g["offset"]) * np.array([1.0 if side == "left" else -1.0, 1.0, 1.0])
        self.base = world.sim.data.qpos.copy()
        self.home = self.base[self.qadr].copy()
        m = self.full.m
        robot = subtree(m, mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, world.env.robots[0].robot_model.root_body))
        obst = set()
        for oid in obstacles:
            obst |= subtree(m, mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, world.objects[oid].root_body))
        self.cm, self.cd = collision_model(world)
        ct0, ca0 = world._mask0
        solid = (ct0 | ca0) > 0
        self.cm.geom_contype[:] = 0
        self.cm.geom_conaffinity[:] = 0
        self.cm.geom_contype[solid & np.isin(self.cm.geom_bodyid, list(robot))] = 2
        if obst:
            self.cm.geom_conaffinity[solid & np.isin(self.cm.geom_bodyid, list(obst))] = 2
        self.any_obst = bool(obst)
        self._place_cache = {}

    @property
    def grasp_z(self):
        return self.g["height"] + self.g["top"] * 2 * self.half[2]

    def edge(self, spin):
        if self.g.get("inset") is None:
            return 0.0
        u = (rot_z(spin) @ self.a0)[:2]
        hx, hy = self.half[:2]
        d = min(hx / max(abs(u[0]), 1e-6), hy / max(abs(u[1]), 1e-6))
        return max(d - self.g["inset"], 0.0)

    def pose(self, k):
        R = rot_z(k.spin) @ self.R0
        a = rot_z(k.spin) @ self.a0
        return k.center - R @ self.offset - (k.back + self.edge(k.spin)) * a + k.up * UP, R

    def solve(self, k, waist, q0=None):
        p, R = self.pose(k)
        base = self.base.copy()
        base[self.qadr[7:]] = waist
        qa, ok = self.arm.solve(p, R, None if q0 is None else q0[:7], base=base)
        return np.concatenate([qa, waist]), ok

    def feasible(self, keys):
        try:
            return self._feasible(keys)
        except Exception as e:
            if _IK_DEVICE[0] == "cpu" or "cuda" not in f"{type(e).__name__} {e}".lower():
                raise
            ik_to_cpu(e)
            return self._feasible(keys)

    def _feasible(self, keys):
        import torch
        G = batch_ik(self.w, self.side)
        poses = [self.pose(Key(k.center, spin, k.back, k.up)) for spin in SPIN for k in keys]
        nw, n = len(WAIST_GRID), len(poses)
        pos = torch.tensor(np.array([p for p, _ in poses]), device=G.device).repeat(nw, 1)
        rot = torch.tensor(np.array([r for _, r in poses]), device=G.device).repeat(nw, 1, 1)
        waist = torch.tensor(WAIST_GRID, device=G.device).repeat_interleave(n, 0)
        _, ok = G.solve_many(pos, rot, waist)
        return ok.reshape(nw, len(SPIN), len(keys)).cpu().numpy()

    def hits(self, q):
        if not self.any_obst:
            return False
        m, d = self.cm, self.cd
        d.qpos[:] = self.base
        d.qpos[self.qadr] = q
        mujoco.mj_kinematics(m, d)
        mujoco.mj_collision(m, d)
        return d.ncon > 0

    def clear(self, steps):
        return not any(self.hits(q) for q, _ in steps[::CHECK_STRIDE])

    def line(self, k0, k1, q, waist, min_steps, name, speed=SPEED):
        p0, _ = self.pose(k0)
        p1, _ = self.pose(k1)
        dist = max(np.linalg.norm(p1 - p0), np.linalg.norm(k1.center - k0.center))
        n = max(min_steps, int(dist / speed * HZ), int(abs(k1.spin - k0.spin) / 0.05))
        out = []
        for s in smooth(n):
            q, ok = self.solve(k0.mix(k1, s), waist, q)
            if not ok:
                raise PlanError(f"ik_{name}")
            out.append(q)
        return out

    def joint(self, q0, q1):
        n = int(np.clip(np.abs(q1 - q0).max() / 0.04, 15, 60))
        return [q0 + s * (q1 - q0) for s in smooth(n)]

    def pick(self, ks, waist):
        pre, grasp, lift = ks
        close = 1.0
        q_pre, _ = self.solve(pre, waist)
        steps = [(q, 0.0) for q in self.joint(self.home, q_pre)]
        steps += [(q, 0.0) for q in self.line(pre, grasp, q_pre, waist, 20, "approach")]
        q = steps[-1][0]
        steps += [(q, g) for g in np.linspace(0.0, close, 15)] + [(q, close)] * SETTLE
        steps += [(q, close) for q in self.line(grasp, lift, q, waist, 20, "lift", SLOW)]
        return steps

    def place_core(self, ks, waist, wi):
        key = (wi, round(ks[0].spin, 6))
        if key not in self._place_cache:
            above, release, retreat = ks
            close = 1.0
            try:
                q_above, ok = self.solve(above, waist)
                if not ok:
                    raise PlanError("ik_above")
                core = [(q, close) for q in self.line(above, release, q_above, waist, 20, "lower", SLOW)]
                q = core[-1][0]
                core += [(q, close)] * SETTLE + [(q, g) for g in np.linspace(close, 0.0, 12)]
                core += [(q, 0.0) for q in self.line(release, retreat, q, waist, 15, "retreat", SLOW)]
                core += [(q, 0.0) for q in self.joint(core[-1][0], self.home)]
                core += [(self.home, 0.0)] * 10
                res = (q_above, core, None if self.clear(core) else "collision")
            except PlanError as e:
                res = (None, None, str(e))
            self._place_cache[key] = res
        return self._place_cache[key]

    def place(self, ks, waist, q, wi):
        q_above, core, err = self.place_core(ks, waist, wi)
        if err:
            raise PlanError(err)
        close = 1.0
        return [(x, close) for x in self.joint(q, q_above)], core

    def first(self, keys, ok, make):
        err = "ik"
        for si in np.flatnonzero(ok):
            ks = [Key(k.center, SPIN[si], k.back, k.up) for k in keys]
            try:
                out = make(ks)
            except PlanError as e:
                err = str(e)
                continue
            if isinstance(out, tuple):
                head, core = out
                if self.clear(head):
                    return head + core, None
            elif self.clear(out):
                return out, None
            err = "collision"
        return None, err

    def plans(self, obj_xy, place_xy, surface_z, carry_z):
        f = self.w.frame
        gz = self.grasp_z
        c0 = np.array([*f.to_world(obj_xy), f.table_z + gz])
        c1 = np.array([*f.to_world(place_xy), surface_z + gz + RELEASE_LIFT])
        hi = carry_z + gz
        pick_keys = [Key(c0, 0, PRE_BACK, PRE_UP), Key(c0, 0), Key([*c0[:2], hi], 0)]
        place_keys = [Key([*c1[:2], hi], 0), Key(c1, 0), Key(c1, 0, 0.0, hi - c1[2])]
        ok = self.feasible(pick_keys + place_keys)
        pick_ok, place_ok = ok[:, :, :3].all(-1), ok[:, :, 3:].all(-1)
        w1, w2 = np.flatnonzero(pick_ok.any(1)), np.flatnonzero(place_ok.any(1))
        if not len(w1) or not len(w2):
            raise PlanError("ik")
        cost = (WAIST_COST[w1][:, None] + WAIST_COST[w2][None] +
                2 * np.abs(WAIST_GRID[w1][:, None] - WAIST_GRID[w2][None]).sum(-1))
        picks, err = {}, "ik"
        for flat in np.argsort(cost, axis=None)[:PAIRS]:
            i, j = w1[flat // len(w2)], w2[flat % len(w2)]
            if i not in picks:
                picks[i] = self.first(pick_keys, pick_ok[i], lambda ks: self.pick(ks, WAIST_GRID[i]))
            first, e = picks[i]
            if first is None:
                err = e
                continue
            second, e = self.first(place_keys, place_ok[j],
                                   lambda ks: self.place(ks, WAIST_GRID[j], first[-1][0], j))
            if second is None:
                err = e
                continue
            yield first + second
        raise PlanError(err)

    def pick_plans(self, obj_xy, carry_z):
        f = self.w.frame
        gz = self.grasp_z
        c0 = np.array([*f.to_world(obj_xy), f.table_z + gz])
        pick_keys = [Key(c0, 0, PRE_BACK, PRE_UP), Key(c0, 0), Key([*c0[:2], carry_z + gz], 0)]
        ok = self.feasible(pick_keys).all(-1)
        w1 = np.flatnonzero(ok.any(1))
        if not len(w1):
            raise PlanError("ik")
        err = "ik"
        for i in w1[np.argsort(WAIST_COST[w1])][:PAIRS]:
            first, e = self.first(pick_keys, ok[i], lambda ks: self.pick(ks, WAIST_GRID[i]))
            if first is None:
                err = e
                continue
            yield first
        raise PlanError(err)


class Recorder:
    def __init__(self, world, side, grasp=None):
        self.w, self.side = world, side
        g = grasp or GRASPS["side"]
        self.open = np.broadcast_to(np.asarray(g.get("open", 0.0), float), (6,)).copy()
        self.close = np.broadcast_to(np.asarray(g.get("close", GRIP_CLOSED), float), (6,)).copy()
        m = world.sim.model
        self.other = [m.get_joint_qpos_addr(n) for n in arm_joints("left" if side == "right" else "right")]
        self.home = world.sim.data.qpos.copy()
        self.actions = []

    def action(self, q, grip):
        a = np.zeros(29)
        arm, hand, _ = SIDES[self.side]
        a[SIDES["left" if self.side == "right" else "right"][0]] = self.home[self.other]
        a[arm] = q[:7]
        a[TORSO] = q[7:10]
        a[hand] = self.open + grip * (self.close - self.open)
        return a

    def execute(self, steps):
        for q, grip in steps:
            self.actions.append(self.action(q, grip).astype(np.float32))
            self.w.step(self.actions[-1])


def surface_of(world, rec):
    ins = rec["instruction"]
    if ins["goal"] == "on_top_of":
        return world.frame.table_z + 2 * world.half_extents(ins["goal_anchor"])[0][2]
    return world.frame.table_z


def carry_height(world, rec):
    top = max(2 * world.half_extents(o["id"])[0][2] for o in rec["objects"] if o["id"] != rec["target"])
    return world.frame.table_z + top + CARRY_CLEAR


def score(world, rec, before):
    ids = [o["id"] for o in rec["objects"]]
    st = world.state(ids)
    tgt, ins = rec["target"], rec["instruction"]
    cands = [c for c in st if st[c]["role"] == "candidate"]
    goal = REL.relation(st, ins["goal"], tgt, [ins["goal_anchor"]], cands, rec["next_to_radius"], world.frame.table_z)
    moved = {o: float(np.hypot(*np.subtract(st[o]["xy"], before[o]["xy"]))) for o in ids if o != tgt}
    worst = max(moved, key=moved.get) if moved else tgt
    out = {"goal": bool(goal["holds"]), "tilt": float(st[tgt]["tilt_deg"]), "disturbed": worst,
           "disturbance": moved.get(worst, 0.0)}
    out["success"] = out["goal"] and out["tilt"] < MAX_TILT and out["disturbance"] < MAX_DISTURB
    return out


def planner_for(world, rec, side, grasp=None):
    tgt = rec["target"]
    return Planner(world, side, [o["id"] for o in rec["objects"] if o["id"] != tgt], grasp or grasp_profile(),
                   world.half_extents(tgt)[0])


def run(world, rec, grasp=None):
    before = world.place(rec["objects"])
    tgt = rec["target"]
    args = (before[tgt]["xy"], rec["place_point"], surface_of(world, rec), carry_height(world, rec))
    order = ("right", "left") if before[tgt]["xy"][1] <= 0 else ("left", "right")
    errors = {}
    for side in order:
        plans = planner_for(world, rec, side, grasp).plans(*args)
        for _ in range(TRIES):
            try:
                steps = next(plans)
            except (PlanError, StopIteration) as e:
                errors[side] = str(e) or errors.get(side, "ik")
                break
            before = world.place(rec["objects"])
            r = Recorder(world, side, grasp or grasp_profile())
            r.execute(steps)
            out = score(world, rec, before)
            out.update(side=side, steps=len(steps))
            if out["success"]:
                return np.stack(r.actions), out
            errors[side] = f"goal {out['goal']} tilt {out['tilt']:.0f} moved {100 * out['disturbance']:.0f}cm"
    return None, {"success": False, "error": errors}


def replay(world, rec, actions, size=None, camera="egoview"):
    from spatialcasa import obs as O
    before = world.place(rec["objects"])
    frames, states, qpos = [], [], []
    for a in actions:
        if size:
            frames.append(O.ego_view(world, size))
        states.append(O.state29(world.env).astype(np.float32))
        qpos.append(world.sim.data.qpos[world._robot_qadr].astype(np.float32))
        world.step(a)
    out = score(world, rec, before)
    ep = {"frames": np.stack(frames) if frames else None, "state": np.stack(states), "qpos": np.stack(qpos),
          "action": np.asarray(actions, np.float32)}
    return ep, out
