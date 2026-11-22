import argparse
import json

import numpy as np

CELL = 0.02
X_RANGE = (0.10, 0.80)
Y_RANGE = (-0.60, 0.60)
CARRY = 0.25


def build(world):
    import torch
    from spatialcasa import demo
    from spatialcasa.gpu_ik import BatchIK
    f = world.frame
    grid = [(i, j) for i in range(int(X_RANGE[0] / CELL), int(X_RANGE[1] / CELL))
            for j in range(int(Y_RANGE[0] / CELL), int(Y_RANGE[1] / CELL))]
    maps = {}
    for side in ("right", "left"):
        P = demo.Planner(world, side)
        G = BatchIK(world, side)
        poses = []
        for i, j in grid:
            c = np.array([*f.to_world(((i + 0.5) * CELL, (j + 0.5) * CELL)), f.table_z + demo.GRASP_HEIGHT])
            for spin in demo.SPIN:
                for k in (demo.Key(c, spin, demo.PRE_BACK, demo.PRE_UP), demo.Key(c, spin),
                          demo.Key([*c[:2], f.table_z + CARRY], spin)):
                    poses.append(P.pose(k))
        pos = torch.tensor(np.array([p for p, _ in poses]), device=G.device)
        rot = torch.tensor(np.array([r for _, r in poses]), device=G.device)
        q, ok = G.solve_many(pos, rot)
        n, s = len(grid), len(demo.SPIN)
        waist = q[:, 7:].reshape(n, s, 3, 3)
        cands = torch.cat([waist, waist.mean(2, keepdim=True)], 2)
        full_ok = ok.reshape(n, s, 3).all(-1)
        W = cands[:, :, :, None].expand(n, s, 4, 3, 3).reshape(-1, 3)
        rep = lambda x: x.reshape(n, s, 1, 3, *x.shape[1:]).expand(n, s, 4, 3, *x.shape[1:]).reshape(-1, *x.shape[1:])
        _, arm_ok = G.solve_many(rep(pos), rep(rot), W)
        good = (arm_ok.reshape(n, s, 4, 3).all(-1).any(-1) & full_ok).any(-1).cpu().numpy()
        maps[side] = [cell for cell, g in zip(grid, good) if g]
    return {"cell": CELL, "x_range": X_RANGE, "y_range": Y_RANGE, "carry": CARRY, "cells": maps}


class Reach:
    def __init__(self, path):
        with open(path) as f:
            r = json.load(f)
        self.cell = r["cell"]
        self.sets = {arm: set(map(tuple, cs)) for arm, cs in r["cells"].items()}

    def key(self, xy):
        return int(np.floor(xy[0] / self.cell)), int(np.floor(xy[1] / self.cell))

    def arms_at(self, xy):
        k = self.key(xy)
        return [arm for arm, s in self.sets.items() if k in s]

    def same_arm(self, a, b):
        return bool(set(self.arms_at(a)) & set(self.arms_at(b)))

    def ok(self, xy, margin=0.0):
        pts = [xy] + ([(xy[0] + dx, xy[1] + dy) for dx, dy in ((margin, 0), (-margin, 0), (0, margin), (0, -margin))]
                      if margin else [])
        return all(self.arms_at(p) for p in pts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="configs/reach_gr1.json")
    a = ap.parse_args()
    from spatialcasa.assets import world_objects
    from spatialcasa.world import World
    r = build(World(world_objects()))
    with open(a.out, "w") as f:
        json.dump(r, f)
    for arm, cs in r["cells"].items():
        c = (np.array(cs) + 0.5) * CELL
        print(arm, len(cs), "cells; x", c[:, 0].min(), c[:, 0].max(), "y", c[:, 1].min(), c[:, 1].max())


if __name__ == "__main__":
    main()
