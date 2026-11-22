import argparse
import json
import os
import shutil
from collections import Counter, defaultdict

import h5py
import numpy as np
import pandas as pd

from export_lerobot import CHUNK, VIDEO_KEY, demo_meta, layout_key, select, task_ids, write_episode, write_meta
from spatialcasa import task as T

SUB_FACTORS = ("background", "clutter", "synonym", "flip", "gap_near", "gap_far", "extra_can", "anchor_move")


def old_episodes(old):
    out = []
    for line in open(os.path.join(old, "meta", "episodes.jsonl")):
        e = json.loads(line)
        ep, chunk = e["episode_index"], e["episode_index"] // CHUNK
        out.append({"kind": "old", "cell": e["cell"], "sub_factor": e["sub_factor"], "meta": e,
                    "parquet": os.path.join(old, "data", f"chunk-{chunk:03d}", f"episode_{ep:06d}.parquet"),
                    "video": os.path.join(old, "videos", f"chunk-{chunk:03d}", VIDEO_KEY, f"episode_{ep:06d}.mp4"),
                    "key": layout_key(e["source"].split(":")[0]) + (e["source"],)})
    return out


def new_episodes(extra):
    out = []
    for p, g in select(extra, None):
        with h5py.File(p) as f:
            a = {k: (v.decode() if isinstance(v, bytes) else v) for k, v in f["data"][g].attrs.items()}
        src = f"{os.path.basename(p)}:{g}"
        out.append({"kind": "new", "cell": a["cell"], "sub_factor": a.get("sub_factor", "clean"), "path": p,
                    "group": g, "attrs": a, "key": layout_key(p) + (src,), "source": src})
    return out


def pick(pool, quota):
    by_cell = defaultdict(list)
    for e in pool:
        by_cell[e["cell"]].append(e)
    queues = [sorted(v, key=lambda e: (e["kind"] != "old", e["key"])) for _, v in sorted(by_cell.items())]
    out = []
    while queues and len(out) < quota:
        for q in queues:
            if q and len(out) < quota:
                out.append(q.pop(0))
        queues = [q for q in queues if q]
    return out


def merge(task, old, extras, out, clean, per_sf):
    pool = old_episodes(old) + [e for x in extras if os.path.isdir(x) for e in new_episodes(x)]
    quotas = {"clean": clean, **{s: per_sf for s in SUB_FACTORS}}
    chosen, short = [], {}
    for k, q in quotas.items():
        got = pick([e for e in pool if e["sub_factor"] == k], q)
        if len(got) < q:
            short[k] = q - len(got)
        chosen += got
    if short:
        return {"task": task, "short": short}
    chosen.sort(key=lambda e: e["key"])
    if os.path.exists(out):
        shutil.rmtree(out)
    os.makedirs(os.path.join(out, "meta"))
    tasks, task_id = task_ids()
    episodes, all_s, all_a, index = [], [], [], 0
    for ep, e in enumerate(chosen):
        if e["kind"] == "old":
            df = pd.read_parquet(e["parquet"])
            state, action = list(df["observation.state"]), list(df["action"])
            m = {k: v for k, v in e["meta"].items() if k not in ("episode_index", "tasks", "length")}
            episodes.append(write_episode(out, ep, index, state, action, e["meta"]["tasks"][1], task_id, m,
                                          video=e["video"]))
        else:
            with h5py.File(e["path"]) as f:
                d = f["data"][e["group"]]
                state, action = d["obs/state"][:], d["actions"][:]
                frames = d["obs/egoview"][:] if "obs/egoview" in d else None
                video = None if frames is not None else os.path.join(os.path.dirname(e["path"]), d.attrs["video"])
            a = e["attrs"]
            episodes.append(write_episode(out, ep, index, state, action, a["instruction"], task_id,
                                          demo_meta(a, e["source"]), frames, video))
        all_s.append(np.stack(state))
        all_a.append(np.stack(action))
        index += len(action)
    write_meta(out, episodes, tasks, index, all_s, all_a)
    return {"task": task, "episodes": len(episodes), "frames": index, "instructions": len(tasks),
            "kinds": dict(Counter(e["sub_factor"] for e in chosen)),
            "new": sum(e["kind"] == "new" for e in chosen)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--old", required=True)
    ap.add_argument("--extra", nargs="+", default=["out/train_extra"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--tasks", nargs="+", default=list(T.GR1_TASKS))
    ap.add_argument("--clean", type=int, default=200)
    ap.add_argument("--per-sub-factor", type=int, default=100)
    a = ap.parse_args()
    for task in a.tasks:
        r = merge(task, os.path.join(a.old, f"gr1_spatial.{task}"), [os.path.join(x, task, "demos") for x in a.extra],
                  os.path.join(a.out, f"gr1_spatial.{task}"), a.clean, a.per_sub_factor)
        print(json.dumps(r), flush=True)


if __name__ == "__main__":
    main()
