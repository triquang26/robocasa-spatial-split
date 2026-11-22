import mujoco
import numpy as np

ARM = ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow_pitch", "wrist_yaw", "wrist_roll", "wrist_pitch")
WAIST = ("waist_yaw", "waist_pitch", "waist_roll")
WAIST_RANGE = {"waist_yaw": (-0.9, 0.9), "waist_pitch": (0.0, 0.9), "waist_roll": (-0.15, 0.15)}


def arm_joints(side):
    return [f"robot0_{side[0]}_{n}" for n in ARM]


def waist_joints():
    return [f"robot0_torso_{n}" for n in WAIST]


class IK:
    def __init__(self, world, side, use_waist=True):
        self.m = world.sim.model._model
        self.d = mujoco.MjData(self.m)
        self.world = world
        self.site = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_SITE, f"gripper0_{side}_grip_site")
        names = arm_joints(side) + (waist_joints() if use_waist else [])
        ids = [mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_JOINT, n) for n in names]
        self.names = names
        self.qadr = np.array([self.m.jnt_qposadr[i] for i in ids])
        self.vadr = np.array([self.m.jnt_dofadr[i] for i in ids])
        lo, hi = self.m.jnt_range[ids].T.copy()
        for k, n in enumerate(names):
            short = n.replace("robot0_torso_", "")
            if short in WAIST_RANGE:
                lo[k], hi[k] = WAIST_RANGE[short]
        self.lo, self.hi = lo, hi
        self.rest = world.sim.data.qpos[self.qadr].copy()

    def fk(self, q, base_qpos=None):
        self.d.qpos[:] = self.world.sim.data.qpos if base_qpos is None else base_qpos
        self.d.qpos[self.qadr] = q
        mujoco.mj_kinematics(self.m, self.d)
        mujoco.mj_comPos(self.m, self.d)
        return self.d.site_xpos[self.site].copy(), self.d.site_xmat[self.site].reshape(3, 3).copy()

    def solve(self, pos, rot, q0=None, base=None, iters=300, damping=0.05, w_rot=0.3, w_rest=0.02, tol=(0.002, 0.03)):
        q = (self.rest if q0 is None else np.asarray(q0, float)).copy()
        base = (self.world.sim.data.qpos if base is None else base).copy()
        jp = np.zeros((3, self.m.nv))
        jr = np.zeros((3, self.m.nv))
        for _ in range(iters):
            p, R = self.fk(q, base)
            ep = pos - p
            er = rot_error(rot, R)
            if np.linalg.norm(ep) < tol[0] and np.linalg.norm(er) < tol[1]:
                return q, True
            mujoco.mj_jacSite(self.m, self.d, jp, jr, self.site)
            J = np.vstack([jp[:, self.vadr], w_rot * jr[:, self.vadr]])
            e = np.concatenate([ep, w_rot * er])
            JJ = J @ J.T + damping ** 2 * np.eye(6)
            r = w_rest * (self.rest - q)
            dq = r + J.T @ np.linalg.solve(JJ, e - J @ r)
            q = np.clip(q + np.clip(dq, -0.2, 0.2), self.lo, self.hi)
        p, R = self.fk(q, base)
        ok = np.linalg.norm(pos - p) < 3 * tol[0] and np.linalg.norm(rot_error(rot, R)) < 3 * tol[1]
        return q, ok


def rot_error(target, current):
    E = target @ current.T
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, E.flatten())
    if q[0] < 0:
        q = -q
    return 2.0 * q[1:]
