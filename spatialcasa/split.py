import json
import os

from spatialcasa import scene as S
from spatialcasa import task as T

EVAL_PER_TASK = 5
EVAL_START = {"eval_seen": 1000, "eval_novel": 2000}
EVAL_LEVELS = ["easy", "medium", "hard"]
DIMENSION = {"background": "VSR", "clutter": "VSR", "synonym": "SSU", "flip": "SSU", "gap_near": "OCD",
             "gap_far": "OCD", "extra_can": "SCR", "anchor_move": "SCR"}


def probe_cells(task, probe_dir):
    res = json.load(open(os.path.join(probe_dir, f"{task}_cells.json")))["cells"]
    order = [c["cell_id"] for c in S.cells()]
    ok = [c for c in order if res.get(c, {}).get("ok")]
    return ([c for c in ok if res[c]["column"] == "train_cells"],
            [c for c in ok if res[c]["column"] == "novel_cells"])


SPARES = 25
SLOT_STRIDE = 100
OVERFLOW_START = 100000
OVERFLOW_STRIDE = 1000


def eval_cells(task, pool, per_task=EVAL_PER_TASK):
    i = list(T.GR1_TASKS).index(task)
    return [pool[(per_task * i + j) % len(pool)] for j in range(per_task)]


def slot_seed(task, name, level, r):
    i = list(T.GR1_TASKS).index(task)
    j = 3 * i + EVAL_LEVELS.index(level)
    if r < SLOT_STRIDE:
        return EVAL_START[name] + SLOT_STRIDE * j + r
    return EVAL_START[name] + OVERFLOW_START + OVERFLOW_STRIDE * j + r - SLOT_STRIDE


def level_slots(task, probe_dir, name, per_task=EVAL_PER_TASK):
    train, novel = probe_cells(task, probe_dir)
    pool = train if name == "eval_seen" else novel
    if not pool:
        return []
    return [{"level": lv, "index": j, "cell": c} for lv in EVAL_LEVELS
            for j, c in enumerate(eval_cells(task, pool, per_task))]


def split_for(task, probe_dir, per_task=EVAL_PER_TASK):
    train, novel = probe_cells(task, probe_dir)
    out = {"task": task, "train_cells": train, "novel_cells": novel}
    for name, pool in (("eval_seen", train), ("eval_novel", novel)):
        out[name + "_cells"] = eval_cells(task, pool, per_task) if pool else []
    return out


def manifest(probe_dir, tasks=None, per_task=EVAL_PER_TASK):
    return {t: split_for(t, probe_dir, per_task) for t in (tasks or T.GR1_TASKS)}


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", default="out/probe/v2")
    ap.add_argument("--per-task", type=int, default=EVAL_PER_TASK)
    a = ap.parse_args()
    print(json.dumps(manifest(a.probe, per_task=a.per_task), indent=1))
