import argparse
import glob
import json
import os
import shutil
import time

import h5py
import numpy as np

from spatialcasa import demo
from spatialcasa import task as T
from spatialcasa.assets import world_objects
from spatialcasa.world import World

LAYOUT = {"right_arm": [0, 7], "left_arm": [7, 14], "waist": [14, 17], "right_hand": [17, 23], "left_hand": [23, 29]}


class Worlds(dict):
    def __init__(self, size):
        super().__init__()
        self.size = size

    def __missing__(self, key):
        style, clutter = key
        self[key] = World(world_objects(with_clutter=clutter), style_id=style, size=self.size)
        return self[key]

    def for_rec(self, rec):
        return self[rec.get("style_id", 0), any(o.get("role") == "clutter" for o in rec["objects"])]


def records(scene_dir, levels, sub_factors, recipes=False):
    if recipes:
        for lv in levels:
            for p in sorted(glob.glob(os.path.join(scene_dir, lv, "*.json"))):
                yield f"{lv}/{os.path.basename(p)[:-5]}", json.load(open(p))
        return
    yield "base", json.load(open(os.path.join(scene_dir, "base.json")))
    for lv in levels:
        for sf in sub_factors:
            p = os.path.join(scene_dir, lv, f"{sf}.json")
            if os.path.exists(p):
                yield f"{lv}/{sf}", json.load(open(p))


def write(group, ep, rec, out, cell):
    group.create_dataset("obs/egoview", data=ep["frames"], compression="gzip", compression_opts=4, chunks=True)
    group.create_dataset("obs/state", data=ep["state"])
    group.create_dataset("obs/qpos", data=ep["qpos"])
    group.create_dataset("actions", data=ep["action"])
    group.attrs["instruction"] = rec["sentence"]
    group.attrs["target"] = rec["target"]
    group.attrs["task"] = rec.get("task", T.current())
    group.attrs["cell"] = cell
    group.attrs["sub_factor"] = rec.get("sub_factor", "clean")
    group.attrs["style_id"] = rec.get("style_id", 0)
    group.attrs["light"] = rec.get("light", 1.0)
    group.attrs["objects"] = json.dumps(rec["objects"])
    group.attrs["score"] = json.dumps(out)
    group.attrs["camera"] = "egoview attached to the head"


def scene_dirs(root):
    return sorted(d for d in glob.glob(os.path.join(root, "*", "s*")) if os.path.exists(os.path.join(d, "base.json")))


def run_scene(W, sdir, a):
    cell, scene = sdir.split(os.sep)[-2:]
    path = os.path.join(a.out, f"{cell}_{scene}.hdf5")
    done = path + ".log.json"
    if a.resume and os.path.exists(done):
        return json.load(open(done))
    tmp = path + ".tmp"
    log = []
    with h5py.File(tmp, "w") as f:
        f.attrs["action_layout"] = json.dumps(LAYOUT)
        f.attrs["state_layout"] = json.dumps(LAYOUT)
        f.attrs["image"] = "egoview rendered 640x400, zero padded to square, resized to 256"
        f.attrs["hz"] = demo.HZ
        f.attrs["task"] = T.current()
        for name, rec in records(sdir, a.levels, a.sub_factors):
            t0 = time.time()
            w = W.for_rec(rec)
            w.set_light(rec.get("light", 1.0))
            try:
                actions, out = demo.run(w, rec)
                t1 = time.time()
                if actions is not None and out["success"]:
                    ep, again = demo.replay(w, rec, actions, a.size)
                    out["replay"] = bool(again["success"])
                    out["success"] = out["replay"]
                    if not out["replay"]:
                        out["error"] = {"replay": f"goal {again['goal']} tilt {again['tilt']:.0f} "
                                                  f"moved {100 * again['disturbance']:.0f}cm"}
                out["plan_sec"] = round(t1 - t0, 1)
            except Exception as e:
                out = {"success": False, "error": {"exception": f"{type(e).__name__}: {e}"}}
            w.set_light(1.0)
            out["sec"] = round(time.time() - t0, 1)
            if out["success"]:
                write(f.create_group(f"data/{name.replace('/', '__')}"), ep, rec, out, cell)
            log.append({"task": T.current(), "cell": cell, "scene": scene, "item": name, **out})
            print(f"{cell} {scene} {name:20s} {'ok' if out['success'] else 'FAIL'} "
                  f"{ {k: v for k, v in out.items() if k != 'success'} }", flush=True)
    if any(r["success"] for r in log):
        os.replace(tmp, path)
    else:
        os.remove(tmp)
    with open(done, "w") as f:
        json.dump(log, f, default=float)
    return log


def oracle_scene(W, sdir, a):
    cell, scene = sdir.split(os.sep)[-2:]
    done = os.path.join(a.out, f"{cell}_{scene}.oracle.json")
    if a.resume and os.path.exists(done):
        return json.load(open(done))
    log = []
    for name, rec in records(sdir, a.levels, a.sub_factors, a.recipes):
        t0 = time.time()
        w = W.for_rec(rec)
        try:
            actions, out = demo.run(w, rec)
            out["steps"] = 0 if actions is None else len(actions)
        except Exception as e:
            out = {"success": False, "error": {"exception": f"{type(e).__name__}: {e}"}}
        out["sec"] = round(time.time() - t0, 1)
        log.append({"task": T.current(), "cell": cell, "scene": scene, "item": name, **out})
        print(f"{cell} {scene} {name:20s} {'ok' if out['success'] else 'FAIL'} {out.get('error', '')}", flush=True)
    with open(done, "w") as f:
        json.dump(log, f, default=float)
    return log


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default=None)
    ap.add_argument("--scenes", default="out/scenes")
    ap.add_argument("--out", default="out/demos")
    ap.add_argument("--levels", nargs="+", default=["train"])
    ap.add_argument("--sub-factors", nargs="+", default=["background", "clutter", "synonym", "flip", "gap_near",
                                                          "gap_far", "extra_can", "anchor_move"])
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--shard", type=int, nargs=2, default=(0, 1))
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--min-free-gb", type=float, default=20.0)
    ap.add_argument("--oracle", action="store_true")
    ap.add_argument("--recipes", action="store_true")
    ap.add_argument("--dirs", nargs="+", default=None)
    a = ap.parse_args()
    T.activate(a.task)
    os.makedirs(a.out, exist_ok=True)
    W = Worlds(a.size)
    step = oracle_scene if a.oracle else run_scene
    log = []
    for sdir in a.dirs or scene_dirs(a.scenes)[a.shard[0]::a.shard[1]]:
        free = shutil.disk_usage(a.out).free / 2 ** 30
        if free < a.min_free_gb:
            print(f"STOP: {free:.0f} GB free < {a.min_free_gb}", flush=True)
            break
        log += step(W, sdir, a)
    if not a.dirs:
        with open(os.path.join(a.out, f"{'oracle' if a.oracle else 'log'}_{a.shard[0]}of{a.shard[1]}.json"), "w") as f:
            json.dump(log, f, indent=1, default=float)
    n = sum(r["success"] for r in log)
    secs = [r["sec"] for r in log if "sec" in r]
    print(f"kept {n}/{len(log)}  mean {np.mean(secs) if secs else 0:.1f} s/item")


if __name__ == "__main__":
    main()
