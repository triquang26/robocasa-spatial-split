import argparse
import json
import os
import time
from collections import Counter

from spatialcasa import scene as S
from spatialcasa import task as T
from spatialcasa.assets import world_objects
from spatialcasa.reach import Reach
from spatialcasa.world import World


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="PlateToBowl")
    ap.add_argument("--reach", default=None)
    ap.add_argument("--tries", type=int, default=S.TRIES)
    ap.add_argument("--cells", nargs="*", default=None)
    ap.add_argument("--out", default="out/probe")
    a = ap.parse_args()
    t = T.activate(a.task)
    L = S.load_yaml("levels.yaml")
    rp = a.reach or T.reach_path(t)
    if not os.path.exists(rp):
        rp = os.path.join(S.HERE, "configs", "reach_gr1.json")
    reach = Reach(rp)
    t0 = time.time()
    w = World(world_objects(with_clutter=False), size=256)
    g = S.Geometry(w)
    print(f"world {time.time() - t0:.1f}s reach={rp}", flush=True)
    res, total = {}, Counter()
    for cell in S.cells():
        if a.cells and cell["cell_id"] not in a.cells:
            continue
        t1 = time.time()
        sc = S.build_base(w, g, reach, cell, 0, L["base_gap"], L["next_to_radius"]["train"],
                          ("probe", a.task, cell["cell_id"]), tries=a.tries)
        ok = "failed" not in sc
        why = {} if ok else sc["failed"]
        total.update(why)
        res[cell["cell_id"]] = {"ok": ok, "column": cell["column"], "sim_tries": sc["sim_tries"], "why": why,
                                "sec": round(time.time() - t1, 2)}
        print(cell["cell_id"], "OK" if ok else "FAIL", sc["sim_tries"], why, f"{time.time() - t1:.1f}s", flush=True)
    n_ok = sum(r["ok"] for r in res.values())
    print(f"ok {n_ok}/{len(res)}  reasons {dict(total.most_common())}  {time.time() - t0:.0f}s")
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, f"{a.task}_cells.json"), "w") as f:
        json.dump({"task": a.task, "reach": rp, "ok": n_ok, "reasons": dict(total), "cells": res}, f, indent=1)


if __name__ == "__main__":
    main()
