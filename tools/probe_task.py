import argparse
import json

import numpy as np

from spatialcasa import relations as REL
from spatialcasa import task as T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--xy", type=float, nargs=2, default=[0.40, 0.0])
    a = ap.parse_args()
    t = T.activate(a.task)
    from spatialcasa.assets import world_objects
    from spatialcasa.world import World
    w = World(world_objects(with_clutter=False, extra=False), render=False)
    print("counter_rect", [round(v, 3) for v in w.counter_rect], "table_z", round(w.frame.table_z, 4))
    out = {"task": a.task, "sizes": {}, "alone": {}, "drop": None}
    for oid in w.objects:
        he, _ = w.half_extents(oid)
        out["sizes"][oid] = [round(float(2 * v), 4) for v in he]
    for oid in [t["ids"][0]] + list(t["anchor_ids"]):
        st = w.place([dict(id=oid, role=w.roles[oid], xy=list(a.xy), yaw=0.0)])[oid]
        out["alone"][oid] = {"drift_mm": round(1000 * float(np.hypot(*np.subtract(st["xy"], a.xy))), 2),
                             "tilt_deg": st["tilt_deg"]}
    main = t["main"]
    c = t["ids"][0]
    objs = [dict(id=main, role="anchor", xy=list(a.xy), yaw=0.0), dict(id=c, role="candidate", xy=list(a.xy), yaw=0.0, on=main)]
    st = w.place(objs, settle=600)
    on = REL.on_top_of(st[c], st[main], w.frame.table_z)
    out["drop"] = {"goal_holds": on["holds"], "tilt_deg": st[c]["tilt_deg"],
                   "drift_mm": round(1000 * float(np.hypot(*np.subtract(st[c]["xy"], a.xy))), 2),
                   "base_above_table_mm": round(1000 * (st[c]["z"] - st[c]["height"] / 2 - w.frame.table_z), 1)}
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
