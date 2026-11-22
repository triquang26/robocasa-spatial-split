import argparse
import importlib
import json
import os
import time
from collections import defaultdict

import cv2
import numpy as np

from spatialcasa import demo
from spatialcasa import obs as O
from spatialcasa.obs import IMAGE, PARTS, RENDER_WH
from spatialcasa import task as T

HORIZON = 720
CHECK_EVERY = 10
REST_SPEED = 0.05


def robot_state(env):
    return {f"state.{k}": v for k, v in O.robot_state(env).items()}


def flat_action(action):
    if isinstance(action, dict):
        a = np.zeros(29)
        for k, (s, e) in PARTS.items():
            a[s:e] = np.asarray(action[f"action.{k}"], float).reshape(-1)[: e - s]
        return a
    a = np.asarray(action, float).reshape(-1)
    assert a.shape == (29,), a.shape
    return a


def split_chunk(action):
    if isinstance(action, dict):
        v = np.asarray(action[f"action.{next(iter(PARTS))}"])
        if v.ndim == 1:
            return [action]
        return [{k: np.asarray(x)[i] for k, x in action.items()} for i in range(v.shape[0])]
    a = np.asarray(action)
    return [a] if a.ndim == 1 else list(a)


class EvalEnv:
    def __init__(self, task, image=IMAGE, horizon=HORIZON):
        from spatialcasa.demos import Worlds
        T.activate(task)
        self.task = task
        self.worlds = Worlds(image)
        self.horizon = horizon
        self.w = self.rec = None

    def _robot_geoms(self, w):
        if not hasattr(w, "_robot_geom_set"):
            m = w.sim.model
            bodies = {b for b in range(m.nbody) if (m.body_id2name(b) or "").startswith(("robot0", "gripper0"))}
            w._robot_geom_set = {g for g in range(m.ngeom) if m.geom_bodyid[g] in bodies}
        return w._robot_geom_set

    def reset(self, rec):
        if self.w is not None:
            self.w.set_light(1.0)
        self.rec = rec
        self.w = self.worlds.for_rec(rec)
        self.w.set_light(rec.get("light", 1.0))
        self.before = self.w.place(rec["objects"])
        self.t = 0
        self.result = None
        return self.obs()

    def obs(self):
        o = {"video.ego_view": O.ego_view(self.w)}
        o.update(robot_state(self.w.env))
        o["annotation.human.task_description"] = self.rec["sentence"]
        o["annotation.human.coarse_action"] = f"unlocked_waist: {self.rec['sentence']}"
        return o

    def check(self):
        w = self.w
        out = demo.score(w, self.rec, self.before)
        tgt = set(w._geoms[self.rec["target"]].tolist())
        robot = self._robot_geoms(w)
        d = w.sim.data
        touch = any((c.geom1 in tgt and c.geom2 in robot) or (c.geom2 in tgt and c.geom1 in robot)
                    for c in d.contact[:d.ncon])
        speed = float(np.linalg.norm(d.get_joint_qvel(w._free_joint(self.rec["target"]))[:3]))
        out.update(released=not touch, speed=speed)
        out["success"] = bool(out["success"] and not touch and speed < REST_SPEED)
        return out

    def step(self, action):
        self.w.step(flat_action(action))
        self.t += 1
        done = False
        if self.t % CHECK_EVERY == 0 or self.t >= self.horizon:
            r = self.check()
            done = r["success"] or self.t >= self.horizon
            if done:
                self.result = dict(r, steps=self.t)
        return self.obs(), done, self.result or {}

    def close(self):
        if self.w is not None:
            self.w.set_light(1.0)


class OraclePolicy:
    needs_env = True

    def reset(self, env, rec):
        actions, out = demo.run(env.w, rec)
        self.queue = [] if actions is None else list(actions)
        self.hold = None if actions is None else actions[-1]

    def __call__(self, obs):
        if self.queue:
            self.hold = self.queue.pop(0)
        return self.hold if self.hold is not None else np.zeros(29)


class Gr00tPolicy:
    def __init__(self, host="localhost", port=5555):
        from gr00t.eval.robot import RobotInferenceClient
        self.client = RobotInferenceClient(host=host, port=port)
        self.queue = []

    def reset(self, env, rec):
        self.queue = []

    def __call__(self, obs):
        if not self.queue:
            batch = {k: (v[None] if isinstance(v, np.ndarray) else [v]) for k, v in obs.items()}
            batch["video.ego_view"] = obs["video.ego_view"][None]
            self.queue = split_chunk({k: np.asarray(v) for k, v in self.client.get_action(batch).items()})
        return self.queue.pop(0)


def make_policy(spec):
    if spec == "oracle":
        return OraclePolicy()
    if spec.startswith("gr00t://"):
        host, _, port = spec[len("gr00t://"):].partition(":")
        return Gr00tPolicy(host, int(port or 5555))
    mod, _, attr = spec.partition(":")
    return getattr(importlib.import_module(mod), attr)()


def load_episodes(root, splits=None, tasks=None, solvable_only=True):
    out = []
    with open(os.path.join(root, "episodes.jsonl")) as f:
        for line in f:
            ep = json.loads(line)
            if splits and ep["split"] not in splits:
                continue
            if tasks and ep["task"] not in tasks:
                continue
            if solvable_only and not ep.get("solvable", True):
                continue
            out.append(ep)
    return out


def rollout(env, policy, ep, rec, video=None):
    obs = env.reset(rec)
    if hasattr(policy, "reset"):
        policy.reset(env, rec)
        if getattr(policy, "needs_env", False):
            obs = env.reset(rec)
    frames = [obs["video.ego_view"]] if video else None
    t0 = time.time()
    while True:
        obs, done, info = env.step(policy(obs))
        if frames is not None:
            frames.append(obs["video.ego_view"])
        if done:
            break
    if video:
        os.makedirs(os.path.dirname(video), exist_ok=True)
        vw = cv2.VideoWriter(video, cv2.VideoWriter_fourcc(*"mp4v"), demo.HZ, (IMAGE, IMAGE))
        for fr in frames:
            vw.write(fr[:, :, ::-1])
        vw.release()
    keys = ("success", "goal", "tilt", "disturbance", "released", "steps")
    return dict({k: ep.get(k) for k in ("episode_id", "split", "task", "cell", "level", "recipe_id", "sub_factor",
                                        "sub_factors", "dimension", "dimensions")},
                **{k: info.get(k) for k in keys}, sec=round(time.time() - t0, 1))


def summarize(rows):
    groups = defaultdict(lambda: [0, 0])
    for r in rows:
        sp = f"split={r['split']}"
        keys = ["all", sp, f"{sp}|level={r['level']}", f"task={r['task']}"]
        for d in r.get("dimensions") or [r["dimension"]]:
            keys += [f"{sp}|dimension={d}", f"{sp}|level={r['level']}|dimension={d}"]
        for sf in r.get("sub_factors") or [r["sub_factor"]]:
            keys.append(f"{sp}|sub_factor={sf}")
        for key in keys:
            groups[key][0] += int(bool(r["success"]))
            groups[key][1] += 1
    return {k: {"success": v[0], "episodes": v[1], "rate": round(v[0] / v[1], 4)} for k, v in sorted(groups.items())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-dir", required=True)
    ap.add_argument("--policy", default="oracle")
    ap.add_argument("--splits", nargs="+", default=None)
    ap.add_argument("--tasks", nargs="+", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--shard", type=int, nargs=2, default=(0, 1))
    ap.add_argument("--videos", type=int, default=0)
    ap.add_argument("--out", default="out/bench_results")
    a = ap.parse_args()
    eps = load_episodes(a.eval_dir, a.splits, a.tasks)[a.shard[0]::a.shard[1]][: a.limit]
    os.makedirs(a.out, exist_ok=True)
    path = os.path.join(a.out, f"results_{a.shard[0]}of{a.shard[1]}.jsonl")
    done = set()
    if os.path.exists(path):
        done = {json.loads(l)["episode_id"] for l in open(path)}
    policy = make_policy(a.policy)
    env = None
    n_video = 0
    with open(path, "a") as f:
        for ep in sorted(eps, key=lambda e: (e["task"], e["episode_id"])):
            if ep["episode_id"] in done:
                continue
            if env is None or env.task != ep["task"]:
                env = EvalEnv(ep["task"])
            rec = json.load(open(os.path.join(a.eval_dir, ep["scene"])))
            video = None
            if n_video < a.videos:
                video = os.path.join(a.out, "videos", ep["episode_id"].replace("/", "__") + ".mp4")
                n_video += 1
            row = rollout(env, policy, ep, rec, video)
            f.write(json.dumps(row) + "\n")
            f.flush()
            print(f"{ep['episode_id']:80s} {'ok' if row['success'] else 'FAIL'} steps {row['steps']}", flush=True)
    rows = []
    for p in os.listdir(a.out):
        if p.startswith("results_") and p.endswith(".jsonl"):
            rows += [json.loads(l) for l in open(os.path.join(a.out, p))]
    s = summarize(rows)
    with open(os.path.join(a.out, "summary.json"), "w") as f:
        json.dump(s, f, indent=1)
    for k, v in s.items():
        print(f"{k:60s} {v['success']:5d}/{v['episodes']:<5d} {100 * v['rate']:.1f}%")


if __name__ == "__main__":
    main()
