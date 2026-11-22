import argparse
import glob
import json
import os
import shutil
import subprocess

import h5py
import numpy as np
import pandas as pd

from spatialcasa import task as T
from spatialcasa.obs import IMAGE, PARTS

FPS = 20
CHUNK = 1000
VIDEO_KEY = "observation.images.ego_view"


def names():
    out = []
    for k, (s, e) in PARTS.items():
        out += [f"{k}_{i}" for i in range(e - s)]
    return out


def encode(frames, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    h, w = frames.shape[1:3]
    cmd = ["ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}",
           "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", "-g", "2", path]
    subprocess.run(cmd, input=np.ascontiguousarray(frames).tobytes(), check=True)


def stats(x):
    x = np.concatenate(x).astype(np.float64)
    return {"mean": x.mean(0).tolist(), "std": x.std(0).tolist(), "min": x.min(0).tolist(),
            "max": x.max(0).tolist(), "q01": np.quantile(x, 0.01, 0).tolist(), "q99": np.quantile(x, 0.99, 0).tolist()}


def layout_key(path):
    name = os.path.basename(path)[:-5]
    cell, _, seed = name.rpartition("_s")
    return int(seed) if seed.isdigit() else 0, cell


def select(src, limit, head_only=True):
    by_cell = {}
    for p in sorted(glob.glob(os.path.join(src, "*.hdf5")), key=layout_key):
        with h5py.File(p) as f:
            keys = sorted(g for g in f["data"] if not head_only or "camera" in f["data"][g].attrs)
        by_cell.setdefault(layout_key(p)[1], []).extend((p, g) for g in keys)
    queues = [list(v) for _, v in sorted(by_cell.items())]
    out = []
    while queues and (limit is None or len(out) < limit):
        for q in queues:
            if q and (limit is None or len(out) < limit):
                out.append(q.pop(0))
        queues = [q for q in queues if q]
    return sorted(out, key=lambda x: (layout_key(x[0]), x[1]))


def demos(src, limit=None):
    for p, g in select(src, limit):
        with h5py.File(p) as f:
            d = f["data"][g]
            if "obs/qpos" not in d:
                raise ValueError(f"{p}:{g} is an old-format demo (square frames, raw qpos state); regenerate it")
            frames = d["obs/egoview"][:] if "obs/egoview" in d else None
            video = None if frames is not None else os.path.join(os.path.dirname(p), d.attrs["video"])
            yield {"frames": frames, "video": video, "state": d["obs/state"][:], "action": d["actions"][:],
                   "attrs": {k: (v.decode() if isinstance(v, bytes) else v) for k, v in d.attrs.items()},
                   "source": f"{os.path.basename(p)}:{g}"}


def write_episode(out, ep, index, state, action, instr, task_id, meta, frames=None, video=None):
    n = len(action)
    coarse = task_id(f"unlocked_waist: {instr}")
    desc = task_id(instr)
    chunk = ep // CHUNK
    df = pd.DataFrame({
        "observation.state": list(np.asarray(state, np.float32)),
        "action": list(np.asarray(action, np.float32)),
        "timestamp": (np.arange(n) / FPS).astype(np.float32),
        "frame_index": np.arange(n, dtype=np.int64),
        "episode_index": np.full(n, ep, np.int64),
        "index": np.arange(index, index + n, dtype=np.int64),
        "task_index": np.full(n, coarse, np.int64),
        "annotation.human.coarse_action": np.full(n, coarse, np.int64),
        "annotation.human.task_description": np.full(n, desc, np.int64),
        "next.done": np.arange(n) == n - 1,
    })
    dp = os.path.join(out, "data", f"chunk-{chunk:03d}", f"episode_{ep:06d}.parquet")
    os.makedirs(os.path.dirname(dp), exist_ok=True)
    df.to_parquet(dp, index=False)
    vp = os.path.join(out, "videos", f"chunk-{chunk:03d}", VIDEO_KEY, f"episode_{ep:06d}.mp4")
    if frames is None:
        os.makedirs(os.path.dirname(vp), exist_ok=True)
        shutil.copyfile(video, vp)
    else:
        encode(frames, vp)
    return dict({"episode_index": ep, "tasks": [f"unlocked_waist: {instr}", instr], "length": n}, **meta)


def write_meta(out, episodes, tasks, index, all_s, all_a):
    with open(os.path.join(out, "meta", "episodes.jsonl"), "w") as f:
        f.writelines(json.dumps(e) + "\n" for e in episodes)
    with open(os.path.join(out, "meta", "tasks.jsonl"), "w") as f:
        f.writelines(json.dumps({"task_index": i, "task": t}) + "\n" for t, i in tasks.items())
    part = {k: {"start": s, "end": e, "absolute": True} for k, (s, e) in PARTS.items()}
    modality = {"state": part, "action": part, "video": {"ego_view": {"original_key": VIDEO_KEY}},
                "annotation": {"human.coarse_action": {}, "human.task_description": {}}}
    vec = {"dtype": "float32", "shape": [29], "names": names()}
    i64 = {"dtype": "int64", "shape": [1], "names": None}
    info = {"codebase_version": "v2.0", "robot_type": "GR1ArmsAndWaistFourierHands",
            "total_episodes": len(episodes), "total_frames": index, "total_tasks": len(tasks),
            "total_videos": len(episodes), "total_chunks": (len(episodes) - 1) // CHUNK + 1, "chunks_size": CHUNK,
            "fps": FPS, "splits": {"train": f"0:{len(episodes)}"},
            "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
            "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
            "features": {VIDEO_KEY: {"dtype": "video", "shape": [IMAGE, IMAGE, 3],
                                     "names": ["height", "width", "channel"],
                                     "video_info": {"video.fps": float(FPS), "video.codec": "h264",
                                                    "video.pix_fmt": "yuv420p", "video.is_depth_map": False,
                                                    "has_audio": False}},
                         "observation.state": vec, "action": vec,
                         "timestamp": {"dtype": "float32", "shape": [1], "names": None},
                         "frame_index": i64, "episode_index": i64, "index": i64, "task_index": i64,
                         "annotation.human.coarse_action": i64, "annotation.human.task_description": i64,
                         "next.done": {"dtype": "bool", "shape": [1], "names": None}}}
    for name, obj in (("info.json", info), ("modality.json", modality),
                      ("stats.json", {"observation.state": stats(all_s), "action": stats(all_a)})):
        with open(os.path.join(out, "meta", name), "w") as f:
            json.dump(obj, f, indent=1)


def task_ids():
    tasks = {}

    def task_id(text):
        if text not in tasks:
            tasks[text] = len(tasks)
        return tasks[text]
    return tasks, task_id


def demo_meta(a, source):
    return {"cell": a["cell"], "sub_factor": a["sub_factor"], "target": a["target"], "style_id": int(a["style_id"]),
            "light": float(a["light"]), "objects": [o["id"] for o in json.loads(a["objects"])], "source": source}


def export_task(task, src, out, limit=None):
    if os.path.exists(out):
        shutil.rmtree(out)
    os.makedirs(os.path.join(out, "meta"))
    tasks, task_id = task_ids()
    episodes, all_s, all_a, index = [], [], [], 0
    for ep, d in enumerate(demos(src, limit)):
        a = d["attrs"]
        episodes.append(write_episode(out, ep, index, d["state"], d["action"], a["instruction"], task_id,
                                      demo_meta(a, d["source"]), d["frames"], d["video"]))
        all_s.append(d["state"])
        all_a.append(d["action"])
        index += len(d["action"])
    if not episodes:
        shutil.rmtree(out)
        return None
    write_meta(out, episodes, tasks, index, all_s, all_a)
    return {"task": task, "episodes": len(episodes), "frames": index, "instructions": len(tasks)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="out/train")
    ap.add_argument("--out", default="out/SpatialCasa-GR1/train")
    ap.add_argument("--tasks", nargs="+", default=list(T.GR1_TASKS))
    ap.add_argument("--per-task", type=int, default=1000)
    a = ap.parse_args()
    res = {}
    for task in a.tasks:
        src = os.path.join(a.src, task, "demos")
        if not os.path.isdir(src):
            src = os.path.join(a.src, task)
        r = export_task(task, src, os.path.join(a.out, f"gr1_spatial.{task}"), a.per_task if a.per_task > 0 else None)
        if r:
            res[task] = r
            print(r, flush=True)
    with open(os.path.join(a.out, "index.json"), "w") as f:
        json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
