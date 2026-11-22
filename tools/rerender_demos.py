import argparse
import glob
import json
import os
import sys
import time

import h5py
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from spatialcasa import demo
from spatialcasa import task as T
from spatialcasa.demos import Worlds, records
from export_lerobot import encode


def scene_records(src, task, fname):
    cell, _, seed = fname[:-5].rpartition("_s")
    sdir = os.path.join(src, task, "scenes", cell, f"s{seed}")
    return {name.replace("/", "__"): rec for name, rec in
            records(sdir, ["train"], ["background", "clutter", "synonym", "flip", "gap_near", "gap_far",
                                      "extra_can", "anchor_move"])}


def video_name(path, group):
    return f"{os.path.basename(path)[:-5]}__{group}.mp4"


def compact_file(path):
    tmp = path + f".{os.getpid()}.rr.tmp"
    with h5py.File(path) as fin, h5py.File(tmp, "w") as fout:
        for k, v in fin.attrs.items():
            fout.attrs[k] = v
        for g in sorted(fin["data"]):
            d = fin["data"][g]
            o = fout.create_group(f"data/{g}")
            for k, v in d.attrs.items():
                o.attrs[k] = v
            if "obs/egoview" in d:
                encode(d["obs/egoview"][:], os.path.join(os.path.dirname(path), video_name(path, g)))
                o.attrs["video"] = video_name(path, g)
            for k in ("obs/state", "obs/qpos", "actions"):
                o.create_dataset(k, data=d[k][:])
    os.replace(tmp, path)


def has_frames(path):
    with h5py.File(path) as f:
        return any("obs/egoview" in f["data"][g] for g in f["data"])


def rerender_file(W, src, task, path):
    recs = scene_records(src, task, os.path.basename(path))
    tmp = path + f".{os.getpid()}.rr.tmp"
    log = []
    with h5py.File(path) as fin, h5py.File(tmp, "w") as fout:
        for k, v in fin.attrs.items():
            fout.attrs[k] = v
        for g in sorted(fin["data"]):
            t0 = time.time()
            d = fin["data"][g]
            actions = d["actions"][:]
            rec = recs[g]
            w = W.for_rec(rec)
            w.set_light(rec.get("light", 1.0))
            ep, score = demo.replay(w, rec, actions, 256)
            w.set_light(1.0)
            diff = float(np.abs(ep["state"] - d["obs/state"][:]).max())
            ok = bool(score["success"])
            log.append({"group": g, "success": ok, "max_state_diff": round(diff, 5), "sec": round(time.time() - t0, 1)})
            if not ok:
                continue
            o = fout.create_group(f"data/{g}")
            for k, v in d.attrs.items():
                o.attrs[k] = v
            o.attrs["camera"] = "egoview attached to the head"
            o.attrs["video"] = video_name(path, g)
            encode(ep["frames"], os.path.join(os.path.dirname(path), video_name(path, g)))
            o.create_dataset("obs/state", data=ep["state"])
            o.create_dataset("obs/qpos", data=ep["qpos"])
            o.create_dataset("actions", data=actions)
    return tmp, log


STALE = 1200


def claim(lock):
    if os.path.exists(lock) and time.time() - os.path.getmtime(lock) > STALE:
        try:
            os.remove(lock)
        except FileNotFoundError:
            pass
    try:
        os.close(os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY))
        return True
    except FileExistsError:
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--src", default="out/train")
    ap.add_argument("--shard", type=int, nargs=2, default=(0, 1))
    ap.add_argument("--needed", type=int, default=None)
    a = ap.parse_args()
    T.activate(a.task)
    demos = os.path.join(a.src, a.task, "demos")
    files = sorted(glob.glob(os.path.join(demos, "*.hdf5")))
    if a.needed:
        from export_lerobot import select
        keep = {p for p, _ in select(demos, a.needed, head_only=False)}
        files = [p for p in files if p in keep]
    files = files[a.shard[0]::a.shard[1]]
    W = Worlds(256)
    for path in files:
        mark = path + ".rr.json"
        lock = path + ".rr.lock"
        if os.path.exists(mark):
            if has_frames(path) and claim(lock):
                compact_file(path)
                os.remove(lock)
                print(os.path.basename(path), "compacted", flush=True)
            continue
        if not claim(lock):
            continue
        if os.path.exists(mark):
            os.remove(lock)
            continue
        tmp, log = rerender_file(W, a.src, a.task, path)
        if any(r["success"] for r in log):
            os.replace(tmp, path)
        else:
            os.remove(tmp)
            os.remove(path)
        with open(mark, "w") as f:
            json.dump(log, f)
        os.remove(lock)
        print(os.path.basename(path), json.dumps(log), flush=True)


if __name__ == "__main__":
    main()
