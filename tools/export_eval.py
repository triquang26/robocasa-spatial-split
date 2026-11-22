import argparse
import glob
import json
import os
import shutil
from collections import Counter, defaultdict

from spatialcasa import bench as B
from spatialcasa import recipes as RC
from spatialcasa import split as SP
from spatialcasa import task as T


def pick_items(sl):
    order = sorted(sl["tried"], key=lambda t: -t["solvable"])
    out = []
    for u in RC.units(sl["plan"]):
        t = next((t for t in order if all(x in t["solved"] for x in u)), None)
        if t is not None:
            out += [(t.get("cell", sl["cell"]), t["seed"], rid) for rid in u]
    return out


def oracle_index(root):
    out = {}
    for p in glob.glob(os.path.join(root, "oracle", "*.oracle.json")):
        for r in json.load(open(p)):
            out[r["cell"], r["scene"], r["item"]] = r
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", nargs="+", default=["out/eval_v3", "out/eval_v3_seen"])
    ap.add_argument("--out", default="out/SpatialCasa-GR1/eval")
    ap.add_argument("--probe", default="out/probe/v2")
    ap.add_argument("--splits", nargs="+", default=["eval_seen", "eval_novel"])
    a = ap.parse_args()
    if os.path.exists(a.out):
        shutil.rmtree(a.out)
    os.makedirs(a.out)
    rows, counts = [], defaultdict(Counter)
    for split, task, src in [(sp, t, x) for sp in a.splits for t in T.GR1_TASKS for x in a.src]:
        if True:
            root = os.path.join(src, split, task)
            if not os.path.isdir(root):
                continue
            orc = oracle_index(root)
            summ = os.path.join(root, "oracle", "summary.json")
            if not os.path.exists(summ):
                continue
            n_task, n_plan = 0, 0
            for sl in json.load(open(summ))["slots_detail"]:
                n_plan += len(sl["plan"])
                picked = pick_items(sl)
                counts[split]["slot_items"] += len(picked)
                counts[split]["slot_items_planned"] += len(sl["plan"])
                counts[split]["bases_used"] += len({(c, k) for c, k, _ in picked})
                lv = sl["level"]
                for cell, seed, rid in picked:
                    ref, goal = cell.split("__")
                    scene = f"s{seed}"
                    sdir = os.path.join(root, "scenes", cell, scene)
                    p = os.path.join(sdir, lv, rid + ".json")
                    o = orc[cell, scene, f"{lv}/{rid}"]
                    rec = json.load(open(p))
                    name = f"{cell}__{scene}__{lv}__{rid}"
                    rel = os.path.join("scenes", split, task, name + ".json")
                    os.makedirs(os.path.join(a.out, os.path.dirname(rel)), exist_ok=True)
                    shutil.copy(p, os.path.join(a.out, rel))
                    prev = None
                    if os.path.exists(p[:-5] + ".png"):
                        prev = os.path.join("previews", split, task, name + ".png")
                        os.makedirs(os.path.join(a.out, os.path.dirname(prev)), exist_ok=True)
                        shutil.copy(p[:-5] + ".png", os.path.join(a.out, prev))
                    counts[split]["exported"] += 1
                    n_task += 1
                    rows.append({"episode_id": f"{split}/{task}/{cell}/{scene}/{lv}/{rid}", "split": split,
                                 "slot": f"{split}/{task}/{sl['cell']}/{lv}",
                                 "task": task, "cell": cell, "reference": ref, "goal": goal, "layout": scene,
                                 "level": lv if rid == RC.CLEAN else rec["level"], "primary": rec.get("primary"),
                                 "recipe_id": rid,
                                 "sub_factor": rec["sub_factor"],
                                 "sub_factors": rec["sub_factors"], "op_levels": rec["op_levels"],
                                 "dimension": rec["dimension"], "dimensions": rec["dimensions"],
                                 "instruction": rec["sentence"], "target": rec["target"],
                                 "old_target": rec["old_target"], "style_id": rec.get("style_id"),
                                 "light": rec.get("light", 1.0), "solvable": True, "oracle_steps": o.get("steps"),
                                 "scene": rel, "preview": prev})
            if n_task < n_plan:
                print(f"SHORT {src} {split} {task}: {n_task} of {n_plan} planned episodes")
    with open(os.path.join(a.out, "episodes.jsonl"), "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    man = {"name": "SpatialCasa-GR1 eval", "robot": "GR1ArmsAndWaistFourierHands", "camera": "egoview",
           "control_hz": 20, "horizon": B.HORIZON,
           "observation": {"video.ego_view": [B.IMAGE, B.IMAGE, 3],
                           **{f"state.{k}": e - s for k, (s, e) in B.PARTS.items()},
                           "annotation.human.task_description": "str"},
           "action": {f"action.{k}": e - s for k, (s, e) in B.PARTS.items()},
           "action_layout_flat29": B.PARTS,
           "success": "goal relation holds, target tilt < 15 deg, other objects moved < 2 cm, target released "
                      "and at rest; checked every 10 steps",
           "levels": ["seen", "easy", "medium_single", "hard_single", "medium", "hard"], "dimensions": SP.DIMENSION,
           "layout": "per task and split: 5 cells x 3 levels (easy, medium, hard), one base scene each; every base "
                     "holds a clean item and one item per sub-factor (the primary): easy = the primary alone at easy, "
                     "medium = the primary with one companion at medium, hard = the primary with two to four "
                     "companions at hard; medium_single and hard_single (L2, L3) = the medium and hard items with "
                     "the companions stripped, on the same base; companions follow the SpatialLIBERO schedule "
                     "(seed 20261103); every item is solved by the scripted planner, and a recipe it cannot solve "
                     "on its base scene is built, with its single, on another base scene of the same cell and level",
           "splits": {s: {"episodes": c["exported"], **dict(c)} for s, c in counts.items()},
           "levels_note": "seen: training layouts at training strength (eval_seen only); medium_single and "
                          "hard_single are the L2 and L3 single-change sets",
           "cells": {t: {k: v for k, v in SP.split_for(t, a.probe).items() if k.endswith("cells")} for t in T.GR1_TASKS}}
    with open(os.path.join(a.out, "manifest.json"), "w") as f:
        json.dump(man, f, indent=1)
    print(json.dumps(man["splits"], indent=1))


if __name__ == "__main__":
    main()
