import math
from copy import deepcopy

import numpy as np

from spatialcasa import relations as REL
from spatialcasa import scene as S

DIMENSION = {"background": "VSR", "clutter": "VSR", "synonym": "SSU", "flip": "SSU", "gap_near": "OCD",
             "gap_far": "OCD", "extra_can": "SCR", "anchor_move": "SCR"}
SUB_FACTORS = tuple(DIMENSION)
UNCHANGED_DRIFT = 0.003
CLUTTER_DRIFT = 0.03
ROBUST = 0.005
CORRIDOR = 0.12
CLUTTER_GAP = 0.03
EXTRA_GAP = 0.02


def settle_item(world, objs, scene, moved):
    st = world.place(objs)
    for o in objs:
        if o.get("on"):
            st[o["id"]]["on"] = o["on"]
    base = {o["id"]: o for o in scene["objects"]}
    for o in objs:
        unchanged = o["id"] in base and o["id"] not in moved and o["role"] != "clutter"
        ref = scene["settled"][o["id"]]["xy"] if unchanged else o["xy"]
        d = np.hypot(*np.subtract(st[o["id"]]["xy"], ref))
        tol = CLUTTER_DRIFT if o["role"] == "clutter" else \
            S.DRIFT if (o["id"] in moved or o["id"] not in base) else UNCHANGED_DRIFT
        if d > tol:
            return None, f"drift:{o['id']}"
        if o["role"] == "clutter":
            o["xy"] = list(st[o["id"]]["xy"])
            o["yaw"] = float(st[o["id"]]["yaw"])
        if not o.get("on") and o["role"] != "clutter" and st[o["id"]]["tilt_deg"] > S.TILT:
            return None, f"tilt:{o['id']}"
    return st, "ok"


def robust_unique(st, ins, tgt, R, table_z):
    cands = [c for c in st if st[c]["role"] == "candidate"]
    for dx, dy in ((ROBUST, 0), (-ROBUST, 0), (0, ROBUST), (0, -ROBUST)):
        s2 = deepcopy(st)
        s2[tgt]["xy"] = [st[tgt]["xy"][0] + dx, st[tgt]["xy"][1] + dy]
        s2[tgt]["box"] = {"lo": [st[tgt]["box"]["lo"][0] + dx, st[tgt]["box"]["lo"][1] + dy],
                          "hi": [st[tgt]["box"]["hi"][0] + dx, st[tgt]["box"]["hi"][1] + dy]}
        r = REL.resolve(s2, ins["reference"], ins["ref_anchors"], cands, R, table_z)
        if r["status"] != "unique" or r["holding"] != [tgt]:
            return False
    return True


def seg_dist(p, a, b):
    p, a, b = (np.asarray(v, float) for v in (p, a, b))
    u = b - a
    t = np.clip((p - a) @ u / max(u @ u, 1e-9), 0, 1)
    return float(np.linalg.norm(p - (a + t * u)))


def item(scene, sf, objs, ins, tgt, **kw):
    out = {"sub_factor": sf, "dimension": DIMENSION[sf], "objects": objs, "instruction": ins, "target": tgt,
           "old_target": scene["target"], "preserves_target": tgt == scene["target"], "style_id": scene.get("style_id", 0), "light": scene.get("light", 1.0),
           "next_to_radius": scene["next_to_radius"], "place_point": scene["place_point"],
           "pools": scene.get("pools", {"ref": "canon", "goal": "canon", "verb": 0})}
    out.update(kw)
    return out


def background(ctx, scene, lv, rng):
    P, L = ctx["pools"], ctx["levels"]["levels"][lv]
    key = L["styles"] if L["styles"] != "AB" else ("A" if rng.random() < 0.5 else "B")
    style = int(rng.choice([x for x in P["styles"][key] if x != scene.get("style_id")]))
    band = L["light"][int(rng.integers(len(L["light"])))]
    light = float(rng.uniform(*band))
    w = ctx["world_for_style"](style)
    w.set_light(light)
    st, why = settle_item(w, scene["objects"], scene, set())
    if st is None:
        w.set_light(1.0)
        return {"failed": why}
    ok, why, vis = S.check_visibility(w, [o["id"] for o in scene["objects"] if o["role"] != "clutter"])
    w.set_light(1.0)
    if not ok:
        return {"failed": why}
    return item(scene, "background", deepcopy(scene["objects"]), scene["instruction"], scene["target"],
                style_id=style, light=light, settled=st, visibility=vis)


def clutter(ctx, scene, lv, rng, tries=400):
    P, L, g, w = ctx["pools"], ctx["levels"]["levels"][lv], ctx["geom"], ctx["world"]
    n = int(rng.integers(L["clutter"]["n"][0], L["clutter"]["n"][1] + 1))
    pool = L["clutter"]["pool"]
    if pool == "AB":
        nb = n // 2 + (n % 2) * int(rng.integers(2))
        cats = list(rng.choice(P["clutter"]["B"], size=nb, replace=False))
        cats += list(rng.choice([c for c in P["clutter"]["A"] if c not in cats], size=n - nb, replace=False))
    else:
        cats = list(rng.choice(P["clutter"][pool], size=n, replace=False))
    base = deepcopy(scene["objects"])
    st0 = scene["settled"]
    tgt, pp = scene["target"], scene["place_point"]
    ga = scene["instruction"]["goal_anchor"]
    lanes = [st0[tgt]["xy"], pp, st0[ga]["xy"]]
    objs = deepcopy(base)
    for cat in cats:
        cid = f"clutter_{cat}"
        for _ in range(tries):
            yaw = float(rng.uniform(-math.pi, math.pi))
            xy = [float(rng.uniform(g.rect[0], g.rect[1])), float(rng.uniform(g.rect[2], g.rect[3]))]
            box = g.box(cid, xy, yaw)
            if not g.on_counter(box):
                continue
            if any(seg_dist(xy, (0.0, 0.0), q) < CORRIDOR for q in lanes):
                continue
            if all(S.boxes_apart(box, g.box(o["id"], o["xy"], o.get("yaw", 0.0)),
                                 CLUTTER_GAP if o["role"] != "clutter" else 0.01) for o in objs):
                objs.append(dict(id=cid, role="clutter", xy=xy, yaw=yaw))
                break
        else:
            return {"failed": f"no_slot:{cat}"}
    st, why = settle_item(w, objs, scene, set())
    if st is None:
        return {"failed": why}
    ok, why, vis = S.check_visibility(w, [o["id"] for o in base])
    if not ok:
        return {"failed": why}
    for oid in (o["id"] for o in base):
        if vis[oid] < 0.7 * scene["visibility"][oid]:
            return {"failed": f"occluded:{oid}"}
    return item(scene, "clutter", objs, scene["instruction"], tgt, settled=st, visibility=vis,
                clutter=[o["id"] for o in objs if o["role"] == "clutter"])


def synonym(ctx, scene, lv, rng):
    pool = ctx["levels"]["levels"][lv]["synonym"]
    ref_pool = goal_pool = "B" if pool == "B" else "A"
    verb = int(rng.integers(1, 3)) if pool == "B" else 0
    return item(scene, "synonym", deepcopy(scene["objects"]), scene["instruction"], scene["target"],
                settled=scene["settled"], pools={"ref": ref_pool, "goal": goal_pool, "verb": verb,
                                                 "seed": int(rng.integers(1 << 30))})


def _flip_edits(ins, kinds):
    ref, anchors, goal, ga = ins["reference"], ins["ref_anchors"], ins["goal"], ins["goal_anchor"]
    out = []
    if "ref_inverse" in kinds and ref in REL.INVERSE:
        out.append(("ref_inverse", dict(ins, reference=REL.INVERSE[ref])))
    if "rel_swap" in kinds:
        for r in REL.REFERENCE:
            if r not in (ref, REL.INVERSE.get(ref)) and r not in ("between", "on_top_of"):
                out.append(("rel_swap", dict(ins, reference=r, ref_anchors=[anchors[0]])))
    if "anchor_change" in kinds and ref != "between":
        for a in S.ANCHORS:
            if a != anchors[0]:
                out.append(("anchor_change", dict(ins, ref_anchors=[a])))
    if "goal_change" in kinds:
        base = [e for e in out]
        for kind, e in base:
            for gg in REL.GOALS:
                if gg != goal:
                    out.append((kind + "+goal", dict(e, goal=gg)))
    return out


def flip(ctx, scene, lv, rng):
    g, reach = ctx["geom"], ctx["reach"]
    st, R = scene["settled"], scene["next_to_radius"]
    cands = [c for c in st if st[c]["role"] == "candidate"]
    edits = _flip_edits(scene["instruction"], ctx["levels"]["levels"][lv]["flip"])
    rng.shuffle(edits)
    for kind, ins in edits:
        r = REL.resolve(st, ins["reference"], ins["ref_anchors"], cands, R, g.table_z)
        if r["status"] != "unique" or r["holding"][0] == scene["target"]:
            continue
        t2 = r["holding"][0]
        if not robust_unique(st, ins, t2, R, g.table_z):
            continue
        gr = REL.relation(st, ins["goal"], t2, [ins["goal_anchor"]], cands, R, g.table_z)
        if gr["holds"] or gr.get("border"):
            continue
        pp = S.place_point(st, g, ins["goal"], ins["goal_anchor"], t2, [o for o in st if o != t2], reach, rng)
        if pp is None:
            continue
        it = item(scene, "flip", deepcopy(scene["objects"]), ins, t2, settled=st, flip_kind=kind, place_point=pp)
        wording = ctx["levels"]["levels"][lv].get("flip_wording", {}).get("goal" if "+goal" in kind else "plain")
        if wording == "B":
            it["pools"] = {"ref": "B", "goal": "B", "verb": int(rng.integers(1, 3)), "seed": int(rng.integers(1 << 30))}
        return it
    return {"failed": "no_flip"}


def gap(ctx, scene, lv, rng, which, tries=1500):
    ins = scene["instruction"]
    if ins["reference"] == "on_top_of":
        return {"failed": "not_applicable"}
    g, reach, w = ctx["geom"], ctx["reach"], ctx["world"]
    L = ctx["levels"]
    gap_bin = L["levels"][lv][which]
    R = L["next_to_radius"][lv] if ins["reference"] == "next_to" else scene["next_to_radius"]
    if ins["reference"] == "next_to" and gap_bin[0] >= R - 0.008:
        R = gap_bin[1] + 0.02
    tgt = scene["target"]
    base = scene["objects"]
    sims = 0
    ax = {o["id"]: {"xy": o["xy"], "yaw": o.get("yaw", 0.0)} for o in base if o["role"] == "anchor"}
    for _ in range(tries):
        objs = deepcopy(base)
        if ins["reference"] == "between":
            moved = set(ins["ref_anchors"])
            t = np.array(scene["settled"][tgt]["xy"], float)
            a0, a1 = (np.array(ax[i]["xy"], float) for i in ins["ref_anchors"])
            th = math.atan2(*(a1 - a0)[::-1]) + rng.uniform(-0.6, 0.6) if rng.random() < 0.5 else \
                rng.uniform(-math.pi, math.pi)
            u = np.array([math.cos(th), math.sin(th)])
            for aid, sgn in zip(ins["ref_anchors"], (-1, 1)):
                he_a = g.he[aid][:2]
                yaw = ax[aid]["yaw"]
                half = abs(u[0]) * (abs(math.cos(yaw)) * he_a[0] + abs(math.sin(yaw)) * he_a[1]) + \
                    abs(u[1]) * (abs(math.sin(yaw)) * he_a[0] + abs(math.cos(yaw)) * he_a[1])
                half_t = abs(u[0]) * g.he[tgt][0] + abs(u[1]) * g.he[tgt][1]
                q = t + sgn * (half + half_t + rng.uniform(*gap_bin)) * u
                for o in objs:
                    if o["id"] == aid:
                        o["xy"] = [float(q[0]), float(q[1])]
        else:
            moved = {tgt}
            p, _ = S._target_xy(rng, g, ins["reference"], ins["ref_anchors"], tgt, ax, gap_bin, R)
            for o in objs:
                if o["id"] == tgt:
                    o["xy"] = [float(p[0]), float(p[1])]
        if not all(g.on_counter(g.box(o["id"], o["xy"], o.get("yaw", 0.0))) for o in objs if o["id"] in moved):
            continue
        st0 = g.state(objs)
        ok, why, _ = S.check_labels(st0, g, ins, tgt, gap_bin, R, reach, rng, settled=False)
        if not ok or not robust_unique(st0, ins, tgt, R, g.table_z):
            continue
        sims += 1
        if sims > 30:
            break
        st, why = settle_item(w, objs, scene, moved)
        if st is None:
            continue
        ok, why, info = S.check_labels(st, g, ins, tgt, gap_bin, R, reach, rng, settled=True)
        if ok and S.check_visibility(w, [o["id"] for o in objs])[0]:
            return item(scene, which, objs, ins, tgt, settled=st, next_to_radius=R, target_gap=info["target_gap"],
                        gap_bin=gap_bin, place_point=info["place_point"])
    return {"failed": "no_gap_layout"}


def _satisfies_none(st, ins, cid, R, table_z):
    others = [c for c in st if st[c]["role"] == "candidate"]
    for tup in REL.anchor_tuples(ins["reference"], list(S.ANCHORS)):
        rec = REL.relation(st, ins["reference"], cid, list(tup), others, R, table_z)
        if rec.get("holds") or rec.get("border"):
            return False
    return True


def extra_can(ctx, scene, lv, rng, attempts=8):
    why = {}
    for _ in range(attempts):
        it = _extra_can(ctx, scene, lv, rng)
        if "failed" not in it:
            return it
        why[it["failed"]] = why.get(it["failed"], 0) + 1
    return {"failed": max(why, key=why.get)}


def _extra_can(ctx, scene, lv, rng, tries=800):
    g, reach, w = ctx["geom"], ctx["reach"], ctx["world"]
    lo, hi = ctx["levels"]["levels"][lv]["extra_cans"]
    n = max(1, int(rng.integers(lo, hi + 1)))
    ins, tgt, R = scene["instruction"], scene["target"], scene["next_to_radius"]
    base = scene["objects"]
    objs = deepcopy(base)
    x0, x1, y0, y1 = S.CAN_REGION
    for cid in S.EXTRA[:n]:
        for _ in range(tries):
            q = [float(rng.uniform(x0, x1)), float(rng.uniform(y0, y1))]
            if not reach.ok(q, 0.02) or not g.in_view(cid, q):
                continue
            box = g.box(cid, q, 0.0)
            if not all(S.boxes_apart(box, g.box(o["id"], o["xy"], o.get("yaw", 0.0)), EXTRA_GAP) for o in objs):
                continue
            trial = objs + [dict(id=cid, role="candidate", xy=q, yaw=0.0)]
            st0 = g.state(trial)
            if _satisfies_none(st0, ins, cid, R, g.table_z):
                objs = trial
                break
        else:
            return {"failed": f"no_slot:{cid}"}
    st, why = settle_item(w, objs, scene, set())
    if st is None:
        return {"failed": why}
    ok, why, info = S.check_labels(st, g, ins, tgt, [0.0, 1.0], R, reach, rng, settled=True)
    if not ok:
        return {"failed": "settled_" + why}
    okv, why, vis = S.check_visibility(w, [o["id"] for o in objs])
    if not okv:
        return {"failed": why}
    return item(scene, "extra_can", objs, ins, tgt, settled=st, visibility=vis,
                extra=[o["id"] for o in objs if o["id"] in S.EXTRA])


def anchor_move(ctx, scene, lv, rng, tries=1500):
    ins = scene["instruction"]
    if ins["reference"] == "on_top_of":
        return {"failed": "not_applicable"}
    g, reach, w = ctx["geom"], ctx["reach"], ctx["world"]
    rule = ctx["levels"]["levels"][lv]["anchor_move"]
    min_disp = rule.get("min_disp", 0.0) if isinstance(rule, dict) else 0.0
    R, tgt = scene["next_to_radius"], scene["target"]
    st0 = scene["settled"]
    base = scene["objects"]
    movers = list(ins["ref_anchors"])
    if rule != "any" and not isinstance(rule, dict) and len(movers) > 1:
        movers = movers[:1]
    others = [c for c in st0 if st0[c]["role"] == "candidate" and c != tgt]
    rng.shuffle(others)
    best, why = [], {}
    bump = lambda k: why.__setitem__(k, why.get(k, 0) + 1)
    gx = np.arange(g.rect[0], g.rect[1], 0.015)
    gy = np.arange(g.rect[2], g.rect[3], 0.015)
    grid = np.stack(np.meshgrid(gx, gy), -1).reshape(-1, 2)
    for t2 in others:
        plan = scene.get("planned_move")
        first = [np.array(plan["xy"])] if plan and plan["t2"] == t2 and plan["anchor"] == movers[0] else []
        order = first + [grid[i] for i in rng.permutation(len(grid))]
        n_try = len(order) if len(movers) == 1 else tries
        for i in range(n_try):
            objs = deepcopy(base)
            moved = {}
            for j, o in enumerate(o for o in objs if o["id"] in movers):
                q = order[i] if j == 0 else grid[rng.integers(len(grid))]
                o["xy"] = [float(q[0]), float(q[1])]
                moved[o["id"]] = float(np.hypot(*(q - np.array(st0[o["id"]]["xy"]))))
            if min(moved.values()) < min_disp:
                bump("min_disp")
                continue
            if not all(g.on_counter(g.box(a, o["xy"], o.get("yaw", 0.0))) for o in objs for a in [o["id"]] if a in moved):
                bump("off_counter")
                continue
            stp = g.state(objs)
            if S.min_pair_gap(stp, list(stp)) < S.PRE_GAP:
                bump("overlap")
                continue
            cands = [c for c in stp if stp[c]["role"] == "candidate"]
            r = REL.resolve(stp, ins["reference"], ins["ref_anchors"], cands, R, g.table_z)
            if r["status"] != "unique" or r["holding"] != [t2] or not robust_unique(stp, ins, t2, R, g.table_z):
                bump("not_unique")
                continue
            gr = REL.relation(stp, ins["goal"], t2, [ins["goal_anchor"]], cands, R, g.table_z)
            if gr["holds"] or gr.get("border"):
                bump("goal_true")
                continue
            if S.place_point(stp, g, ins["goal"], ins["goal_anchor"], t2, [o for o in stp if o != t2], reach,
                             rng) is None:
                bump("no_place_point")
                continue
            best.append((sum(moved.values()), t2, objs, moved))
            if len(best) >= 40:
                break
        if len(best) >= 40:
            break
    best.sort(key=lambda b: b[0])
    for _, t2, objs, moved in best:
        st, why = settle_item(w, objs, scene, set(moved))
        if st is None:
            continue
        cands = [c for c in st if st[c]["role"] == "candidate"]
        r = REL.resolve(st, ins["reference"], ins["ref_anchors"], cands, R, g.table_z)
        if r["status"] == "unique" and r["holding"] == [t2]:
            pp = S.place_point(st, g, ins["goal"], ins["goal_anchor"], t2, [o for o in st if o != t2], reach, rng)
            if pp is not None:
                return item(scene, "anchor_move", objs, ins, t2, settled=st, moved=moved, place_point=pp)
    return {"failed": "no_geometry " + str(why) if not best else "settle"}


OPS = {"background": background, "clutter": clutter, "synonym": synonym, "flip": flip,
       "gap_near": lambda c, s, l, r: gap(c, s, l, r, "gap_near"),
       "gap_far": lambda c, s, l, r: gap(c, s, l, r, "gap_far"),
       "extra_can": extra_can, "anchor_move": anchor_move}
