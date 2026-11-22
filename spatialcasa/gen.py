import argparse
import json
import os
import time

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from spatialcasa import operators as OP
from spatialcasa import recipes as RC
from spatialcasa import scene as S
from spatialcasa import task as T
from spatialcasa.assets import world_objects
from spatialcasa.reach import Reach
from spatialcasa.world import World

IMG = 384
OWN_SCENE = ("gap_near", "gap_far", "anchor_move")
SINGLE_RETRIES = 3
ALT_BASES = 6


def jsonable(x):
    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    if isinstance(x, np.ndarray):
        return x.tolist()
    return x


class Ctx(dict):
    def __init__(self, task=None, render=True, reach=None):
        super().__init__()
        t = T.activate(task)
        self.render = render
        self["task"] = t
        self["pools"] = S.load_yaml("pools.yaml")
        self["levels"] = S.load_yaml("levels.yaml")
        self["assets"] = t
        rp = reach or T.reach_path(t)
        if not os.path.exists(rp):
            raise FileNotFoundError(f"reach map {rp} missing; build it with python -m spatialcasa.reach --task {t['name']}")
        self["reach"] = Reach(rp)
        self._worlds = {}
        self["world"] = self.world_for_style(0)
        self["geom"] = S.Geometry(self["world"])
        self["world_for_style"] = self.world_for_style

    def world_for_style(self, style):
        if style not in self._worlds:
            self._worlds[style] = World(world_objects(), style_id=style, size=IMG, render=self.render)
        return self._worlds[style]


def sentence_of(ctx, it_or_scene):
    p = it_or_scene.get("pools", {"ref": "canon", "goal": "canon", "verb": 0})
    rng = np.random.default_rng(p.get("seed", 0)) if p["ref"] != "canon" else None
    nouns = ctx["task"]["candidate"]["nouns"]
    pool = nouns.get(p["ref"], nouns["A"]) if p["ref"] != "canon" else nouns["A"]
    noun = pool[int(np.random.default_rng(p.get("seed", 0) + 1).integers(len(pool)))] if len(pool) > 1 else pool[0]
    return S.sentence(it_or_scene["instruction"], ctx["assets"], ctx["pools"]["lexicon"], p["ref"], p["goal"],
                      p["verb"], noun=noun, rng=rng)


def frame(ctx, rec, style=0, light=1.0):
    w = ctx.world_for_style(style)
    w.set_light(light)
    w.place(rec["objects"])
    img = Image.fromarray(w.render("egoview", IMG))
    w.set_light(1.0)
    d = ImageDraw.Draw(img)
    rings = [(rec["target"], (0, 230, 0))]
    if rec.get("old_target") and rec["old_target"] != rec["target"]:
        rings.append((rec["old_target"], (255, 60, 60)))
    for oid, col in rings:
        s = rec["settled"][oid]
        wxy = w.frame.to_world(s["xy"])
        u, v = w.project([wxy[0], wxy[1], s["z"]], size=IMG)
        r = 22
        d.ellipse([u - r, v - r, u + r, v + r], outline=col, width=3)
    return img


def sheet(rows, path, labels):
    font = ImageFont.load_default()
    cw, chh = IMG // 2, IMG // 2
    H = 18
    ncol = max(len(r[1]) for r in rows)
    out = Image.new("RGB", (90 + ncol * cw, len(rows) * (chh + H) + H), "white")
    d = ImageDraw.Draw(out)
    for j, lab in enumerate(labels):
        d.text((90 + j * cw + 4, 2), lab, fill="black", font=font)
    for i, (name, imgs) in enumerate(rows):
        y = H + i * (chh + H)
        d.text((4, y + chh // 2), name, fill="black", font=font)
        for j, (im, cap) in enumerate(imgs):
            if im is not None:
                out.paste(im.resize((cw, chh)), (90 + j * cw, y))
            d.text((90 + j * cw + 2, y + chh + 2), cap[:34], fill="black", font=font)
    out.save(path)


def base_scene(ctx, cell, key, plan_move=None):
    L = ctx["levels"]
    args = (ctx["world"], ctx["geom"], ctx["reach"], cell)
    for k, extra in ((key, {}), (key + ("fb",), {"allow_flags": ("anchor_blind", "nearest_rule")})):
        sc = S.build_base(*args, key[2] if isinstance(key[2], int) else 0, L["base_gap"], L["next_to_radius"]["train"],
                          k, plan_move=plan_move, **extra)
        if "failed" not in sc:
            sc["old_target"] = sc["target"]
            sc["gap_bin_used"] = L["base_gap"]
            sc["sentence"] = sentence_of(ctx, sc)
            return sc
    return sc


def apply(ctx, sc, cell, cid, k, lv, sf):
    it = OP.OPS[sf](ctx, sc, lv, S.rng_for("item", cid, k, lv, sf))
    if "failed" not in it or sf not in OWN_SCENE:
        return it
    for i in range(ALT_BASES):
        rule = ctx["levels"]["levels"][lv]["anchor_move"]
        plan = (rule.get("min_disp", 0.0) if isinstance(rule, dict) else 0.0) if sf == "anchor_move" else None
        alt = base_scene(ctx, cell, ("base", cid, k, "alt", sf, lv, i), plan_move=plan)
        if "failed" in alt:
            continue
        it2 = OP.OPS[sf](ctx, alt, lv, S.rng_for("item", cid, k, lv, sf, i))
        if "failed" not in it2:
            it2["own_base"] = {"objects": alt["objects"], "settled": alt["settled"], "key": i}
            return it2
    return it


def pick_cells(task, n, probe_dir="out/probe"):
    p = os.path.join(probe_dir, f"{task}_cells.json")
    if not os.path.exists(p):
        raise FileNotFoundError(f"{p} missing; run tools/probe_cells.py --task {task}")
    res = json.load(open(p))["cells"]
    ok = [c for c, r in res.items() if r["ok"]]
    out, refs, goals = [], set(), set()
    while len(out) < n and len(out) < len(ok):
        def gain(c):
            r, g = c.split("__")
            return (r not in refs) + (g not in goals), res[c]["column"] != "novel_cells", -ok.index(c)
        c = max((c for c in ok if c not in out), key=gain)
        out.append(c)
        refs.add(c.split("__")[0])
        goals.add(c.split("__")[1])
    return out


def scene_jobs(cells, layouts, start=0):
    return [(cid, k) for k in range(start, start + layouts) for cid in cells]


def make_scene(ctx, cell, cid, k, a, styles):
    d = os.path.join(a.out, cid, f"s{k}")
    if a.resume and os.path.exists(os.path.join(d, "done.json")):
        return "skip"
    os.makedirs(d, exist_ok=True)
    t0 = time.time()
    sc = base_scene(ctx, cell, ("base", ctx["task"]["name"], cid, k))
    if "failed" in sc:
        with open(os.path.join(d, "done.json"), "w") as f:
            json.dump({"base": "failed", "why": jsonable(sc["failed"])}, f)
        print(cid, k, "base FAILED", sc["failed"], flush=True)
        return "failed"
    sc["style_id"] = int(styles[k % len(styles)])
    sc["task"] = ctx["task"]["name"]
    sc["jitter"] = S.jitter_states(ctx["world"], ctx["geom"], ctx["reach"], sc, n=a.jitter) if a.jitter else []
    with open(os.path.join(d, "base.json"), "w") as f:
        json.dump(jsonable(sc), f, indent=1)
    base_img = frame(ctx, sc, sc["style_id"]) if a.images else None
    if base_img is not None:
        base_img.save(os.path.join(d, "base.png"))
    print(f"{cid} s{k} style {sc['style_id']}: {sc['sentence']!r} flags={sc['flags']}", flush=True)
    rows, items = [], {}
    for lv in a.levels:
        os.makedirs(os.path.join(d, lv), exist_ok=True)
        imgs = [(base_img, "clean")]
        for sf in a.sub_factors:
            it = apply(ctx, sc, cell, cid, k, lv, sf)
            if "failed" in it:
                print(f"  {lv:6s} {sf:11s} FAILED {it['failed']}", flush=True)
                items[f"{lv}/{sf}"] = "failed:" + str(it["failed"])
                imgs.append((None, f"{sf}: {it['failed']}"))
                continue
            if sf != "background":
                it["style_id"] = sc["style_id"]
            it["task"] = sc["task"]
            it["sentence"] = sentence_of(ctx, it)
            with open(os.path.join(d, lv, f"{sf}.json"), "w") as f:
                json.dump(jsonable(it), f, indent=1)
            items[f"{lv}/{sf}"] = "ok"
            if a.images:
                im = frame(ctx, it, it["style_id"], it["light"])
                im.save(os.path.join(d, lv, f"{sf}.png"))
                imgs.append((im, sf + (" *" if "own_base" in it else "")))
            print(f"  {lv:6s} {sf:11s} {it['sentence']!r}", flush=True)
        rows.append((lv, imgs))
    if a.images:
        sheet(rows, os.path.join(d, "sheet.png"), ["clean"] + list(a.sub_factors))
    with open(os.path.join(d, "done.json"), "w") as f:
        json.dump({"base": "ok", "items": items, "sec": round(time.time() - t0, 1)}, f)
    print(f"  {time.time() - t0:.0f} s", flush=True)
    return "ok"


def make_level_scene(ctx, cell, cid, k, lv, index, a, styles):
    d = os.path.join(a.out, cid, f"s{k}")
    if a.resume and os.path.exists(os.path.join(d, "done.json")):
        return "skip"
    os.makedirs(os.path.join(d, lv), exist_ok=True)
    t0 = time.time()
    task = ctx["task"]["name"]
    if lv == "seen":
        sc = json.load(open(os.path.join(a.seen_src[task, cid, k], task, "scenes", cid, f"s{k}", "base.json")))
    else:
        sc = base_scene(ctx, cell, ("base", task, cid, k))
    if "failed" in sc:
        with open(os.path.join(d, "done.json"), "w") as f:
            json.dump({"base": "failed", "level": lv, "index": index, "why": jsonable(sc["failed"])}, f)
        print(cid, k, lv, "base FAILED", sc["failed"], flush=True)
        return "failed"
    if lv != "seen":
        sc["style_id"] = int(styles[k % len(styles)])
    sc["task"] = task
    sc["level"] = lv
    with open(os.path.join(d, "base.json"), "w") as f:
        json.dump(jsonable(sc), f, indent=1)
    if a.images:
        frame(ctx, sc, sc["style_id"]).save(os.path.join(d, "base.png"))
    ids = a.plans[f"{cid}|{lv}"]
    want = a.needs.get(f"{cid}|s{k}") if a.needs else None
    plan = [RC.recipe_of(r) for r in ids if want is None or r in want]

    def alt_base(i, sf, op_lv):
        rule = ctx["levels"]["levels"][op_lv]["anchor_move"]
        pm = (rule.get("min_disp", 0.0) if isinstance(rule, dict) else 0.0) if sf == "anchor_move" else None
        return base_scene(ctx, cell, ("base", task, cid, k, "alt", sf, op_lv, i), plan_move=pm)

    items = {}
    print(f"{cid} s{k} {lv} style {sc['style_id']}: {sc['sentence']!r}", flush=True)
    for r in plan:
        for attempt in range(SINGLE_RETRIES if r["level"] in RC.SINGLE.values() else 1):
            it = RC.build(ctx, sc, r, lv, ("recipe", task, cid, k, lv) + ((attempt,) if attempt else ()), alt_base)
            if "failed" not in it:
                break
        if "failed" in it:
            items[r["id"]] = "failed:" + it["failed"]
            print(f"  {r['id']:50s} FAILED {it['failed']}", flush=True)
            continue
        it["planned"] = r["id"]
        it["sentence"] = sentence_of(ctx, it)
        with open(os.path.join(d, lv, f"{it['recipe_id']}.json"), "w") as f:
            json.dump(jsonable(it), f, indent=1)
        if a.images:
            frame(ctx, it, it["style_id"], it["light"]).save(os.path.join(d, lv, f"{it['recipe_id']}.png"))
        items[r["id"]] = "ok"
        print(f"  {it['recipe_id']:50s} {it['sentence']!r}", flush=True)
    with open(os.path.join(d, "done.json"), "w") as f:
        json.dump({"base": "ok", "level": lv, "index": index, "plan": [r["id"] for r in plan], "items": items,
                   "sec": round(time.time() - t0, 1)}, f)
    print(f"  {time.time() - t0:.0f} s", flush=True)
    return "ok"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default=None)
    ap.add_argument("--cells", nargs="+", default=None)
    ap.add_argument("--jobs", nargs="+", default=None)
    ap.add_argument("--level-jobs", nargs="+", default=None)
    ap.add_argument("--plans", default=None)
    ap.add_argument("--needs", default=None)
    ap.add_argument("--seen-layouts", default=None)
    ap.add_argument("--n-cells", type=int, default=8)
    ap.add_argument("--probe", default="out/probe")
    ap.add_argument("--scenes", "--layouts", dest="layouts", type=int, default=1)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--styles", default="A")
    ap.add_argument("--levels", nargs="+", default=["train"])
    ap.add_argument("--sub-factors", nargs="+", default=list(OP.SUB_FACTORS))
    ap.add_argument("--jitter", type=int, default=0)
    ap.add_argument("--images", action="store_true")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--shard", type=int, nargs=2, default=(0, 1))
    ap.add_argument("--reach", default=None)
    ap.add_argument("--out", default="out/scenes")
    a = ap.parse_args()
    ctx = Ctx(a.task, render=True, reach=a.reach)
    pools = ctx["pools"]["styles"]
    styles = pools[a.styles] if a.styles in pools else [int(x) for x in a.styles.split(",")]
    cells = {c["cell_id"]: c for c in S.cells()}
    if a.level_jobs:
        os.makedirs(a.out, exist_ok=True)
        a.plans = json.load(open(a.plans))
        a.needs = json.load(open(a.needs)) if a.needs else None
        a.seen_src = {}
        for row in (json.load(open(a.seen_layouts)) if a.seen_layouts else []):
            t, c, k, root = row.split("|")
            a.seen_src[t, c, int(k)] = root
        stat = {}
        for j in a.level_jobs[a.shard[0]::a.shard[1]]:
            cid, k, lv, index = j.split(":")
            r = make_level_scene(ctx, cells[cid], cid, int(k), lv, int(index), a, styles)
            stat[r] = stat.get(r, 0) + 1
        print("scenes", stat)
        return
    given = [(j.rsplit(":", 1)[0], int(j.rsplit(":", 1)[1])) for j in a.jobs] if a.jobs else None
    chosen = sorted({c for c, _ in given}) if given else a.cells or pick_cells(ctx["task"]["name"], a.n_cells, a.probe)
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, f"cells_{a.shard[0]}.json"), "w") as f:
        json.dump({"task": ctx["task"]["name"], "cells": chosen, "layouts": a.layouts, "start": a.start,
                   "styles": styles, "levels": a.levels}, f)
    jobs = (given or scene_jobs(chosen, a.layouts, a.start))[a.shard[0]::a.shard[1]]
    stat = {}
    for cid, k in jobs:
        r = make_scene(ctx, cells[cid], cid, k, a, styles)
        stat[r] = stat.get(r, 0) + 1
    print("scenes", stat)


if __name__ == "__main__":
    main()
