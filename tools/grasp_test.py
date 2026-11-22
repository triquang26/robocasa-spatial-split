import argparse
import itertools
import json
import os
import time
from collections import Counter

import numpy as np

from spatialcasa import demo
from spatialcasa import scene as S
from spatialcasa import task as T
from spatialcasa.assets import world_objects
from spatialcasa.reach import Reach
from spatialcasa.world import World

LIFT_MIN = 0.03


def positions(reach, n, seed, region):
    rng = np.random.default_rng(seed)
    x0, x1, y0, y1 = region
    out = []
    while len(out) < n:
        q = [float(rng.uniform(x0, x1)), float(rng.uniform(y0, y1))]
        if reach is None or reach.ok(q, 0.02):
            out.append(q)
    return out


def trial(w, oid, xy, g, hold=10):
    objs = [dict(id=oid, role="candidate", xy=xy, yaw=0.0)]
    before = w.place(objs)
    z0 = before[oid]["z"]
    order = ("right", "left") if xy[1] <= 0 else ("left", "right")
    errs = {}
    for side in order:
        P = demo.Planner(w, side, (), g, w.half_extents(oid)[0])
        carry = w.frame.table_z + 0.15
        try:
            plans = P.pick_plans(before[oid]["xy"], carry)
            steps = next(plans)
        except (demo.PlanError, StopIteration) as e:
            errs[side] = str(e) or "ik"
            continue
        w.place(objs)
        r = demo.Recorder(w, side, g)
        r.execute(steps + [steps[-1]] * hold)
        s = w.state([oid])[oid]
        rise, tilt = s["z"] - z0, s["tilt_deg"]
        if rise >= LIFT_MIN and tilt < demo.MAX_TILT:
            return {"ok": True, "side": side, "rise": round(rise, 4), "tilt": round(tilt, 2)}
        errs[side] = "dropped" if rise < LIFT_MIN else "tilt"
        return {"ok": False, "side": side, "why": errs[side], "rise": round(rise, 4), "tilt": round(tilt, 2)}
    return {"ok": False, "why": "plan:" + ",".join(sorted(set(errs.values()))), "errors": errs}


def grid(spec):
    keys = list(spec)
    for vals in itertools.product(*(spec[k] for k in keys)):
        yield dict(zip(keys, vals))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="PlateToBowl")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--grasp", default=None)
    ap.add_argument("--params", default=None)
    ap.add_argument("--sweep", default=None)
    ap.add_argument("--shard", type=int, nargs=2, default=(0, 1))
    ap.add_argument("--reach", default=None)
    ap.add_argument("--out", default="out/grasp")
    a = ap.parse_args()
    t = T.activate(a.task)
    oid = t["ids"][0]
    base = demo.grasp_profile(a.grasp, json.loads(a.params) if a.params else None)
    rp = a.reach or T.reach_path(t)
    reach = Reach(rp) if os.path.exists(rp) else None
    w = World([o for o in world_objects(with_clutter=False) if o["id"] == oid], render=False)
    pts = positions(reach, a.n, a.seed, S.CAN_REGION)
    configs = [dict(base, **c) for c in grid(json.loads(a.sweep))] if a.sweep else [base]
    configs = configs[a.shard[0]::a.shard[1]]
    os.makedirs(a.out, exist_ok=True)
    rows = []
    for g in configs:
        t0 = time.time()
        res = [trial(w, oid, xy, g) for xy in pts]
        n_ok = sum(r["ok"] for r in res)
        why = Counter(r.get("why") for r in res if not r["ok"])
        row = {"grasp": g, "ok": n_ok, "n": len(res), "why": dict(why), "sec": round(time.time() - t0, 1),
               "trials": [dict(r, xy=xy) for r, xy in zip(res, pts)]}
        rows.append(row)
        print(json.dumps({k: g[k] for k in ("top", "inset", "pitch", "roll", "open", "close")}), f"{n_ok}/{len(res)}",
              dict(why), f"{row['sec']}s", flush=True)
    tag = f"{a.task}_{a.shard[0]}of{a.shard[1]}"
    with open(os.path.join(a.out, f"{tag}.json"), "w") as f:
        json.dump({"task": a.task, "object": oid, "reach": rp if reach else None, "rows": rows}, f, indent=1)


if __name__ == "__main__":
    main()
