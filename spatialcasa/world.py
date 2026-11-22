import json
import math
import os
from collections import OrderedDict
from copy import deepcopy

import mujoco

import numpy as np

import robocasa
from robocasa.environments.tabletop.tabletop import Tabletop
from robocasa.models.fixtures import FixtureType
from robocasa.utils.gym_utils.gymnasium_basic import create_env_robosuite
from robosuite.controllers import load_composite_controller_config

from spatialcasa.frame import Frame, quat_wxyz_to_yaw, yaw_to_quat_wxyz

ROBOT = "GR1ArmsAndWaistFourierHands"
CAMERAS = ("egoview",)
SETTLE_STEPS = 300
DROP = 0.003
OBJ_ROOT = os.path.join(robocasa.models.assets_root, "objects")


class SpatialCounter(Tabletop):

    VALID_LAYOUTS = [0]

    def __init__(self, spec_objects=None, *args, **kwargs):
        self.spec_objects = deepcopy(spec_objects or [])
        kwargs.setdefault("use_distractors", False)
        kwargs.setdefault("distractor_config", None)
        super().__init__(*args, **kwargs)

    def _setup_table_references(self):
        super()._setup_table_references()
        self.counter = self.register_fixture_ref("counter", dict(id=FixtureType.COUNTER, size=(0.45, 0.55)))
        self.init_robot_base_pos = self.counter

    def _get_obj_cfgs(self):
        n = len(self.spec_objects)
        cols = max(1, math.ceil(math.sqrt(n * 2)))
        cfgs = []
        for i, o in enumerate(self.spec_objects):
            r, c = divmod(i, cols)
            pos = (-0.9 + 1.8 * (c + 0.5) / cols, -0.9 + 1.8 * (r + 0.5) / max(1, math.ceil(n / cols)))
            cfgs.append(dict(
                name=o["id"],
                obj_groups=os.path.join(OBJ_ROOT, o["mjcf"]),
                object_scale=o.get("scale"),
                placement=dict(fixture=self.counter, size=(0.05, 0.05), pos=pos,
                               ensure_object_boundary_in_range=False, ensure_valid_placement=False,
                               rotation=(0.0, 0.0))))
        return cfgs

    def get_ep_meta(self):
        m = super().get_ep_meta()
        m["lang"] = ""
        return m

    def _check_success(self):
        return False


def make_env(spec_objects, style_id=0, seed=0, size=256, cameras=CAMERAS, render=True):
    cc = load_composite_controller_config(controller=None, robot=ROBOT)
    cc["type"] = "BASIC"
    cc["composite_controller_specific_configs"] = {}
    cc["control_delta"] = False
    import robosuite
    env = robosuite.make(
        env_name="SpatialCounter", spec_objects=spec_objects, robots=[ROBOT], controller_configs=cc,
        camera_names=list(cameras), camera_widths=size, camera_heights=size, has_renderer=False,
        has_offscreen_renderer=render, ignore_done=True, use_object_obs=False, use_camera_obs=False,
        camera_depths=False, seed=seed, translucent_robot=False, layout_and_style_ids=[(0, int(style_id))])
    return env


CACHE = 256


class Stepper:
    def __init__(self, env):
        from robosuite.controllers.parts.generic.joint_pos import JointPositionController
        assert env.lite_physics
        self.env = env
        self.m, self.d = env.sim.model._model, env.sim.data._data
        self.n = int(env.control_timestep / env.model_timestep)
        r = env.robots[0]
        cc = r.composite_controller
        self.M = np.ndarray((self.m.nv, self.m.nv), dtype=np.float64, order="C")
        self.arms, self.grips = [], []
        for name, c in cc.part_controllers.items():
            if not r._enabled_parts.get(name, False):
                continue
            s, e = cc._action_split_indexes[name]
            act = r._ref_actuators_indexes_dict[name]
            lim = (self.m.actuator_ctrlrange[act, 0].copy(), self.m.actuator_ctrlrange[act, 1].copy(),
                   self.m.actuator_gear[act, 0].copy())
            if name in cc.grippers:
                self.grips.append((s, e, act, lim, c, cc.grippers[name]))
                continue
            assert type(c) is JointPositionController and c.input_type == "absolute" and c.interpolator is None
            vi = np.asarray(c.qvel_index)
            self.arms.append((s, e, act, lim, np.asarray(c.qpos_index), vi, np.array(c.kp),
                              np.array(c.kd)))

    def __call__(self, action):
        m, d = self.m, self.d
        for s, e, act, (lo, hi, gear), c, g in self.grips:
            c.set_goal(g.format_action(action[s:e]))
            d.ctrl[act] = np.clip(c.run_controller() / gear, lo, hi)
        for _ in range(self.n):
            mujoco.mj_step1(m, d)
            mujoco.mj_fullM(m, self.M, d.qM)
            for s, e, act, (lo, hi, gear), qi, vi, kp, kd in self.arms:
                err = np.array(action[s:e]) - np.array(d.qpos[qi])
                tau = np.dot(self.M[vi, :][:, vi], np.multiply(err, kp) + np.multiply(-np.array(d.qvel[vi]), kd))
                d.ctrl[act] = np.clip((tau + d.qfrc_bias[vi]) / gear, lo, hi)
            mujoco.mj_step2(m, d)
        self.env.timestep += 1
        self.env.cur_time += self.env.control_timestep


class World:
    def __init__(self, spec_objects, style_id=0, seed=0, size=256, light_scale=1.0, render=True):
        self.env = make_env(spec_objects, style_id=style_id, seed=seed, size=size, render=render)
        self.env.reset()
        for name in list(self.env._observables):
            self.env.modify_observable(name, "enabled", False)
        self.style_id = int(style_id)
        self.sim = self.env.sim
        self.size = size
        m = self.sim.model
        r = self.env.robots[0]
        bid = m.body_name2id(r.robot_model.root_body)
        base = self.sim.data.body_xpos[bid]
        c = self.env.counter
        self.frame = Frame(base[:2], quat_wxyz_to_yaw(self.sim.data.body_xquat[bid]), float(c.pos[2] + c.size[2] / 2))
        self.counter_rect = self._counter_rect(c)
        self.objects = {o["id"]: self.env.objects[o["id"]] for o in spec_objects}
        self.roles = {o["id"]: o.get("role") for o in spec_objects}
        self._robot_q = self.sim.data.qpos.copy()
        self._robot_qadr = self._robot_qpos_addrs()
        self._light0 = m.light_diffuse.copy()
        self.set_light(light_scale)
        self._geoms = self._object_geoms()
        self._mask0 = (m.geom_contype.copy(), m.geom_conaffinity.copy())
        self._cache = OrderedDict()
        self._spec = mujoco.mjtState.mjSTATE_INTEGRATION
        self._stepper = None


    def _counter_rect(self, c):
        hx, hy = c.size[0] / 2, c.size[1] / 2
        corners = [self.frame.to_robot(np.array(c.pos[:2]) + np.array([sx * hx, sy * hy]))
                   for sx in (-1, 1) for sy in (-1, 1)]
        p = np.array(corners)
        return float(p[:, 0].min()), float(p[:, 0].max()), float(p[:, 1].min()), float(p[:, 1].max())

    def half_extents(self, oid):
        o = self.objects[oid]
        b = np.array(o.get_bbox_points())
        return (b.max(0) - b.min(0)) / 2, float(b[:, 2].min())

    def _robot_qpos_addrs(self):
        m = self.sim.model
        qadr, vadr = [], []
        for j in range(m.njnt):
            n = m.joint_id2name(j) or ""
            if n.startswith(("robot0", "gripper0")) and m.jnt_type[j] in (2, 3):
                qadr.append(m.jnt_qposadr[j])
                vadr.append(m.jnt_dofadr[j])
        self._robot_vadr = np.array(vadr, int)
        return np.array(qadr, int)


    def _object_geoms(self):
        m = self.sim.model
        out = {}
        for oid, o in self.objects.items():
            root = m.body_name2id(o.root_body)
            out[oid] = np.flatnonzero([self._is_desc(b, root) for b in m.geom_bodyid])
        return out

    def _collide_only(self, active):
        m = self.sim.model
        ct, ca = self._mask0
        for oid, g in self._geoms.items():
            on = oid in active
            m.geom_contype[g] = ct[g] if on else 0
            m.geom_conaffinity[g] = ca[g] if on else 0

    def _key(self, spec_objects, settle):
        return json.dumps([[o["id"], [round(float(v), 6) for v in o["xy"]], round(float(o.get("yaw", 0.0)), 6),
                            o.get("on")] for o in spec_objects] + [settle])

    def _free_joint(self, oid):
        return self.objects[oid].joints[0]

    def write_pose(self, oid, xy, yaw, lift=DROP):
        he, bottom = self.half_extents(oid)
        wxy = self.frame.to_world(xy)
        z = self.frame.table_z - bottom + lift
        q = yaw_to_quat_wxyz(self.frame.yaw_to_world(yaw))
        self.sim.data.set_joint_qpos(self._free_joint(oid), np.concatenate([[wxy[0], wxy[1], z], q]))
        self.sim.data.set_joint_qvel(self._free_joint(oid), np.zeros(6))

    def park(self, oid):
        k = list(self.objects).index(oid)
        self.sim.data.set_joint_qpos(self._free_joint(oid), np.array([0.5 * k, 0, -5.0, 1, 0, 0, 0]))
        self.sim.data.set_joint_qvel(self._free_joint(oid), np.zeros(6))

    def support_height(self, spec_by_id, o):
        a = o.get("on")
        return 0.0 if a is None else 2 * float(self.half_extents(a)[0][2])

    def place(self, spec_objects, settle=SETTLE_STEPS):
        by_id = {o["id"]: o for o in spec_objects}
        self._collide_only(by_id)
        key = self._key(spec_objects, settle)
        m, d = self.sim.model._model, self.sim.data._data
        hit = self._cache.get(key)
        if hit is not None:
            self._cache.move_to_end(key)
            mujoco.mj_setState(m, d, hit, self._spec)
            self.sim.forward()
            return self.state(list(by_id))
        self._settle(spec_objects, settle)
        st = np.empty(mujoco.mj_stateSize(m, self._spec))
        mujoco.mj_getState(m, d, st, self._spec)
        self._cache[key] = st
        if len(self._cache) > CACHE:
            self._cache.popitem(last=False)
        return self.state(list(by_id))

    def _settle(self, spec_objects, settle):
        self.sim.data.qpos[self._robot_qadr] = self._robot_q[self._robot_qadr]
        self.sim.data.qvel[:] = 0
        by_id = {o["id"]: o for o in spec_objects}
        for o in spec_objects:
            self.write_pose(o["id"], o["xy"], o.get("yaw", 0.0), lift=DROP + self.support_height(by_id, o))
        parked = [oid for oid in self.objects if oid not in by_id]
        for oid in parked:
            self.park(oid)
        self.sim.forward()
        for _ in range(settle):
            self.sim.data.qpos[self._robot_qadr] = self._robot_q[self._robot_qadr]
            self.sim.data.qvel[self._robot_vadr] = 0
            for oid in parked:
                self.park(oid)
            self.sim.step()
        self.sim.forward()

    def penetration(self):
        d = self.sim.data
        worst = 0.0
        for i in range(d.ncon):
            worst = max(worst, -float(d.contact[i].dist))
        return worst


    def state(self, ids=None):
        ids = ids or list(self.objects)
        out = {}
        for oid in ids:
            o = self.objects[oid]
            b = self.sim.model.body_name2id(o.root_body)
            p = self.sim.data.body_xpos[b]
            q = self.sim.data.body_xquat[b]
            R = self.sim.data.body_xmat[b].reshape(3, 3)
            yaw = self.frame.yaw_to_robot(quat_wxyz_to_yaw(q))
            tilt = math.degrees(math.acos(max(-1.0, min(1.0, R[2, 2]))))
            he, _ = self.half_extents(oid)
            xy = self.frame.to_robot(p[:2])
            c, s = abs(math.cos(yaw)), abs(math.sin(yaw))
            hx, hy = c * he[0] + s * he[1], s * he[0] + c * he[1]
            out[oid] = {"xy": [round(float(xy[0]), 5), round(float(xy[1]), 5)], "yaw": round(float(yaw), 5),
                        "z": round(float(p[2]), 5), "tilt_deg": round(tilt, 3), "role": self.roles.get(oid),
                        "box": {"lo": [float(xy[0] - hx), float(xy[1] - hy)], "hi": [float(xy[0] + hx), float(xy[1] + hy)]},
                        "height": float(2 * he[2])}
        return out


    def step(self, action):
        if self._stepper is None:
            self._stepper = Stepper(self.env)
        self._stepper(action)

    def set_light(self, scale):
        self.sim.model.light_diffuse[:] = np.clip(self._light0 * float(scale), 0, 1)

    def render(self, camera="egoview", size=None, segmentation=False, width=None, height=None):
        s = size or self.size
        self.sim._render_context_offscreen.gl_ctx.make_current()
        img = self.sim.render(width=width or s, height=height or s, camera_name=camera, segmentation=segmentation)
        return img[::-1]

    def project(self, xyz_world, camera="egoview", size=256):
        from robosuite.utils.camera_utils import get_camera_transform_matrix
        M = get_camera_transform_matrix(self.sim, camera, size, size)
        p = M @ np.array([*xyz_world, 1.0])
        return float(p[0] / p[2]), float(p[1] / p[2])

    def in_frame(self, oid, camera="egoview", size=256, margin=2):
        s = self.state([oid])[oid]
        z0, z1 = s["z"] - s["height"] / 2, s["z"] + s["height"] / 2
        for x in (s["box"]["lo"][0], s["box"]["hi"][0]):
            for y in (s["box"]["lo"][1], s["box"]["hi"][1]):
                wx, wy = self.frame.to_world([x, y])
                for z in (z0, z1):
                    u, v = self.project([wx, wy, z], camera, size)
                    if not (margin <= u <= size - margin and margin <= v <= size - margin):
                        return False
        return True

    def visibility(self, camera="egoview", size=256):
        seg = self.render(camera, size, segmentation=True)
        m = self.sim.model
        geom_ids = seg[..., 1][seg[..., 0] == 5]
        bodies = m.geom_bodyid[geom_ids]
        out = {}
        for oid, o in self.objects.items():
            root = m.body_name2id(o.root_body)
            sub = [b for b in range(m.nbody) if self._is_desc(b, root)]
            out[oid] = int(np.isin(bodies, sub).sum())
        return out

    def _is_desc(self, b, root):
        m = self.sim.model
        while b > 0:
            if b == root:
                return True
            b = m.body_parentid[b]
        return False

    def close(self):
        self.env.close()
