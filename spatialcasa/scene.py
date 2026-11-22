import math
import zlib
from copy import deepcopy

import numpy as np
import yaml

from spatialcasa import relations as REL

HERE = __file__.rsplit("/", 2)[0]
ANCHORS = ("plate", "bowl", "board")
CANS = ("can_1", "can_2", "can_3")
EXTRA = ("can_4", "can_5")
MAIN = None
TASK = None
PRE_GAP = 0.007
POST_GAP = 0.005
FIX_GAP = 0.02
DRIFT = 0.005
TILT = 5.0
MIN_PIX = 200
MIN_PIX_ANCHOR = 150
CAN_REGION = (0.20, 0.56, -0.45, 0.45)
REF_REGION = (0.12, 0.66, -0.52, 0.52)
PLACE_GAP = (0.02, 0.10)
PLACE_CLEAR = 0.02
TRIES = 4000


def load_yaml(name):
    with open(f"{HERE}/configs/{name}") as f:
        return yaml.safe_load(f)


def rng_for(*key):
    return np.random.default_rng(zlib.crc32("/".join(map(str, key)).encode()))


def cells():
    held = {("left_of", "left_of"), ("right_of", "right_of"), ("in_front_of", "in_front_of"), ("behind", "behind"),
            ("left_of", "in_front_of"), ("right_of", "behind"), ("in_front_of", "right_of"), ("behind", "left_of"),
            ("nearest_to", "behind"), ("next_to", "left_of"), ("between", "right_of"), ("on_top_of", "in_front_of")}
    out = []
    for r in REL.REFERENCE:
        for g in REL.GOALS:
            out.append({"cell_id": f"{r}__{g}", "reference": r, "goal": g,
                        "column": "novel_cells" if (r, g) in held else "train_cells"})
    return out


def is_candidate(oid):
    return oid in CANS or oid in EXTRA


def quarter_yaw(aid):
    if TASK is None:
        return aid == "board"
    return TASK["anchors"].get(aid, {}).get("yaw") == "quarter"


def anchor_assignment(cell, k):
    if MAIN:
        rest = [x for x in ANCHORS if x != MAIN]
        a, b = rest[k % len(rest)], rest[(k + 1) % len(rest)]
        if cell["reference"] == "between":
            return ([a, b] if a != b else [a, MAIN]), MAIN
        if cell["reference"] == "on_top_of":
            return [a], MAIN
        same_ok = cell["goal"] != cell["reference"]
        return ([MAIN] if (k % 3 == 0 and same_ok) else [a]), MAIN
    perms = [(a, b, c) for a in ANCHORS for b in ANCHORS for c in ANCHORS if len({a, b, c}) == 3]
    a, b, c = perms[k % 6]
    if cell["reference"] == "between":
        return [a, b], c if k % 2 == 0 else a
    if cell["reference"] == "on_top_of":
        return [a], b
    same_ok = cell["goal"] != cell["reference"]
    return [a], a if (k % 3 == 0 and same_ok) else b


class Geometry:

    def __init__(self, world):
        self.he = {oid: world.half_extents(oid)[0] for oid in world.objects}
        self.table_z = world.frame.table_z
        self.rect = world.counter_rect
        self.reach = None
        from robosuite.utils.camera_utils import get_camera_transform_matrix
        self.cam = get_camera_transform_matrix(world.sim, "egoview", 256, 256)
        self.frame = world.frame

    def in_view(self, oid, xy, yaw=0.0, margin=2, base_z=None):
        b = self.box(oid, xy, yaw)
        z0 = self.table_z if base_z is None else base_z
        for x in (b["lo"][0], b["hi"][0]):
            for y in (b["lo"][1], b["hi"][1]):
                wx, wy = self.frame.to_world([x, y])
                for z in (z0, z0 + 2 * self.he[oid][2]):
                    p = self.cam @ np.array([wx, wy, z, 1.0])
                    u, v = p[0] / p[2], p[1] / p[2]
                    if not (margin <= u <= 256 - margin and margin <= v <= 256 - margin):
                        return False
        return True

    def box(self, oid, xy, yaw):
        hx0, hy0 = self.he[oid][:2]
        c, s = abs(math.cos(yaw)), abs(math.sin(yaw))
        hx, hy = c * hx0 + s * hy0, s * hx0 + c * hy0
        return {"lo": [xy[0] - hx, xy[1] - hy], "hi": [xy[0] + hx, xy[1] + hy]}

    def state(self, objs):
        st = {}
        for o in objs:
            base = self.table_z + (2 * self.he[o["on"]][2] if o.get("on") else 0.0)
            st[o["id"]] = {"xy": list(o["xy"]), "yaw": o.get("yaw", 0.0), "box": self.box(o["id"], o["xy"], o.get("yaw", 0.0)),
                           "z": base + self.he[o["id"]][2], "height": 2 * self.he[o["id"]][2], "tilt_deg": 0.0,
                           "role": o.get("role"), "on": o.get("on")}
        return st

    def on_counter(self, box, margin=FIX_GAP):
        x0, x1, y0, y1 = self.rect
        return box["lo"][0] > x0 + margin and box["hi"][0] < x1 - margin and box["lo"][1] > y0 + margin and \
            box["hi"][1] < y1 - margin


def boxes_apart(a, b, g):
    return REL.gap({"box": a}, {"box": b}) >= g


def min_pair_gap(st, ids, skip_on=True):
    worst = math.inf
    ids = list(ids)
    for i, a in enumerate(ids):
        for b in ids[i + 1:]:
            if skip_on and (st[a].get("on") == b or st[b].get("on") == a):
                continue
            worst = min(worst, REL.gap(st[a], st[b]))
    return worst


def _target_xy(rng, g, ref, anchors, tgt, ax, gap_bin, R):
    he_t = g.he[tgt]
    if ref in REL.AXIS:
        k, s = REL.AXIS[ref]
        a = ax[anchors[0]]
        abox = g.box(anchors[0], a["xy"], a["yaw"])
        gap = rng.uniform(*gap_bin)
        half_a = (abox["hi"][k] - abox["lo"][k]) / 2
        disp = half_a + gap + he_t[k]
        lim = disp - math.sqrt(2) * (REL.CFG["eps_border"] + 0.004)
        orth = rng.uniform(-lim, lim) if lim > 0 else 0.0
        p = np.array(a["xy"], float)
        p[k] += s * disp
        p[1 - k] += orth
        return p, None
    if ref in ("nearest_to", "next_to"):
        a = ax[anchors[0]]
        lo, hi = gap_bin
        if ref == "next_to":
            hi = min(hi, R - 0.008)
        gap = rng.uniform(lo, hi)
        th = rng.uniform(-math.pi, math.pi)
        u = np.array([math.cos(th), math.sin(th)])
        c = np.array(a["xy"], float)
        abox = g.box(anchors[0], a["xy"], a["yaw"])
        lo_d, hi_d = 0.0, 1.0
        for _ in range(40):
            mid = (lo_d + hi_d) / 2
            p = c + mid * u
            if REL.gap({"box": g.box(tgt, p, 0.0)}, {"box": abox}) < gap:
                lo_d = mid
            else:
                hi_d = mid
        return c + hi_d * u, None
    if ref == "between":
        a, b = (np.array(ax[i]["xy"], float) for i in anchors)
        t = rng.uniform(0.33, 0.67)
        n = np.array([-(b - a)[1], (b - a)[0]]) / np.linalg.norm(b - a)
        return a + t * (b - a) + rng.uniform(-0.035, 0.035) * n, None
    if ref == "on_top_of":
        a = ax[anchors[0]]
        abox = g.box(anchors[0], a["xy"], a["yaw"])
        ins = 0.025
        lo = np.array(abox["lo"]) + max(ins, 0.0)
        hi = np.array(abox["hi"]) - max(ins, 0.0)
        if np.any(hi <= lo):
            return np.array(a["xy"], float), anchors[0]
        return rng.uniform(lo, hi), anchors[0]
    raise ValueError(ref)


def place_point(st, g, goal, goal_anchor, tgt, others, reach, rng, tries=200):
    if goal == "on_top_of":
        p = list(st[goal_anchor]["xy"])
        if any(o.get("on") == goal_anchor for o in st.values()) or (reach and not reach.same_arm(st[tgt]["xy"], p)):
            return None
        return p
    k, s = REL.AXIS[goal]
    a = st[goal_anchor]
    for _ in range(tries):
        gap = rng.uniform(*PLACE_GAP)
        half_a = (a["box"]["hi"][k] - a["box"]["lo"][k]) / 2
        disp = half_a + gap + g.he[tgt][k]
        lim = disp - math.sqrt(2) * 0.012
        p = np.array(a["xy"], float)
        p[k] += s * disp
        p[1 - k] += rng.uniform(-lim, lim) * 0.7
        box = g.box(tgt, p, 0.0)
        if not g.on_counter(box) or (reach and not reach.same_arm(st[tgt]["xy"], p)):
            continue
        if all(boxes_apart(box, st[o]["box"], PLACE_CLEAR) for o in others):
            if REL.directional({"xy": p, "box": box}, a, goal)["holds"]:
                return [float(p[0]), float(p[1])]
    return None


def instruction_parts(cell, anchors, goal_anchor):
    return {"reference": cell["reference"], "ref_anchors": list(anchors), "goal": cell["goal"],
            "goal_anchor": goal_anchor}


def propose(rng, g, cell, anchors, goal_anchor, gap_bin, R, reach, tgt=None, plan_move=None, plan=None):
    tgt = tgt or CANS[0]
    t2 = CANS[1]
    ax = {}
    for aid in ANCHORS:
        for _ in range(50):
            yaw = float(rng.choice([0.0, math.pi / 2])) if quarter_yaw(aid) else float(rng.uniform(-math.pi, math.pi))
            x0, x1, y0, y1 = REF_REGION if aid in anchors else g.rect
            xy = [float(rng.uniform(x0, x1)), float(rng.uniform(y0, y1))]
            box = g.box(aid, xy, yaw)
            if g.on_counter(box) and all(boxes_apart(box, g.box(b, ax[b]["xy"], ax[b]["yaw"]), 0.02) for b in ax):
                ax[aid] = {"xy": xy, "yaw": yaw}
                break
        else:
            return _why(plan, "anchor:" + aid)
    if cell["reference"] == "between":
        d = np.linalg.norm(np.subtract(ax[anchors[0]]["xy"], ax[anchors[1]]["xy"]))
        if not (0.25 <= d <= 0.50):
            return _why(plan, "between_dist")
    p, on = _target_xy(rng, g, cell["reference"], anchors, tgt, ax, gap_bin, R)
    objs = [dict(id=a, role="anchor", xy=v["xy"], yaw=v["yaw"]) for a, v in ax.items()]
    t = dict(id=tgt, role="candidate", xy=[float(p[0]), float(p[1])], yaw=0.0)
    if on:
        t["on"] = on
    objs.append(t)
    if plan_move is not None:
        a0 = anchors[0]
        for _ in range(60):
            x0, x1, y0, y1 = REF_REGION
            q = [float(rng.uniform(x0, x1)), float(rng.uniform(y0, y1))]
            box = g.box(a0, q, ax[a0]["yaw"])
            if np.hypot(q[0] - ax[a0]["xy"][0], q[1] - ax[a0]["xy"][1]) < max(plan_move, 0.10) or \
                    not g.on_counter(box):
                continue
            if not all(boxes_apart(box, g.box(o["id"], o["xy"], o.get("yaw", 0.0)), 0.02)
                       for o in objs if o["id"] != a0):
                continue
            ax2 = dict(ax, **{a0: {"xy": q, "yaw": ax[a0]["yaw"]}})
            p2, _ = _target_xy(rng, g, cell["reference"], anchors, t2, ax2, gap_bin, R)
            b2 = g.box(t2, p2, 0.0)
            if (reach and not reach.ok(p2, 0.02)) or not boxes_apart(b2, box, PRE_GAP) or \
                    not all(boxes_apart(b2, g.box(o["id"], o["xy"], o.get("yaw", 0.0)), PRE_GAP) for o in objs):
                continue
            objs.append(dict(id=t2, role="candidate", xy=[float(p2[0]), float(p2[1])], yaw=0.0))
            plan["move"] = {"anchor": a0, "xy": q, "t2": t2}
            break
        else:
            return _why(plan, "move")
    x0, x1, y0, y1 = CAN_REGION
    for cid in CANS:
        if cid == tgt or any(o["id"] == cid for o in objs):
            continue
        for _ in range(60):
            q = [float(rng.uniform(x0, x1)), float(rng.uniform(y0, y1))]
            box = g.box(cid, q, 0.0)
            if (reach and not reach.ok(q, 0.02)) or not g.in_view(cid, q):
                continue
            if all(boxes_apart(box, g.box(o["id"], o["xy"], o.get("yaw", 0.0)), PRE_GAP) for o in objs):
                objs.append(dict(id=cid, role="candidate", xy=q, yaw=0.0))
                break
        else:
            return _why(plan, "distractor")
    return objs


def _why(plan, w):
    if plan is not None:
        plan["why"] = w
    return None


def check_labels(st, g, ins, tgt, gap_bin, R, reach, rng, settled):
    ref, anchors, goal, ga = ins["reference"], ins["ref_anchors"], ins["goal"], ins["goal_anchor"]
    cands = [c for c in st if st[c]["role"] == "candidate"]
    ids = list(st)
    if min_pair_gap(st, ids) < (POST_GAP if settled else PRE_GAP):
        return False, "gap_objects", {}
    for c in cands:
        if not g.on_counter(st[c]["box"], 0.0) or (reach and not reach.ok(st[c]["xy"])):
            return False, "reach", {}
        if not g.in_view(c, st[c]["xy"], base_z=st[c]["z"] - st[c]["height"] / 2):
            return False, "view", {}
    r = REL.resolve(st, ref, anchors, cands, R, g.table_z)
    if r["status"] != "unique" or r["holding"] != [tgt]:
        return False, f"reference_{r['status']}", {}
    tg = REL.target_gap(st, ref, tgt, anchors)
    if tg is not None and not (gap_bin[0] <= tg <= gap_bin[1]):
        return False, "gap_bin", {}
    gr = REL.relation(st, goal, tgt, [ga], cands, R, g.table_z)
    if gr["holds"] or gr.get("border"):
        return False, "goal_true", {}
    others = [o for o in ids if o != tgt]
    pp = place_point(st, g, goal, ga, tgt, others, reach, rng)
    if pp is None:
        return False, "no_place_point", {}
    return True, "ok", {"target_gap": tg, "place_point": pp}


def check_physics(st, objs):
    for o in objs:
        s = st[o["id"]]
        if np.hypot(*(np.subtract(s["xy"], o["xy"]))) > DRIFT:
            return False, f"drift:{o['id']}"
        if not o.get("on") and s["tilt_deg"] > TILT:
            return False, f"tilt:{o['id']}"
    return True, "ok"


def check_visibility(world, ids):
    vis = world.visibility()
    for oid in ids:
        if is_candidate(oid) and not world.in_frame(oid):
            return False, f"out_of_frame:{oid}", vis
        need = MIN_PIX if is_candidate(oid) else MIN_PIX_ANCHOR
        if vis.get(oid, 0) < need:
            return False, f"visibility:{oid}", vis
    return True, "ok", vis


def phrase(table, rel, a, b=None, assets=None, pool="canon", rng=None, exclude=()):
    e = table[rel]
    opts = [e["canon"]] if pool == "canon" else [x for x in e[pool] if x not in exclude]
    s = opts[int(rng.integers(len(opts)))] if rng is not None and len(opts) > 1 else opts[0]
    noun = lambda k: assets["anchors"][k]["noun"]
    on = assets["anchors"][a]["goal_word"] if a in assets["anchors"] else "on"
    s = s.replace("{on}", on).replace("onto", "into" if on == "in" else "onto")
    return s.format(a=noun(a), b=noun(b) if b else "")


def sentence(ins, assets, lex, ref_pool="canon", goal_pool="canon", verb=0, noun=None, rng=None):
    noun = noun or (TASK["candidate"]["noun"] if TASK else "can")
    ref = phrase(lex["reference"], ins["reference"], *ins["ref_anchors"][:2], assets=assets, pool=ref_pool, rng=rng) \
        if ins["reference"] == "between" else \
        phrase(lex["reference"], ins["reference"], ins["ref_anchors"][0], assets=assets, pool=ref_pool, rng=rng)
    goal = phrase(lex["goal"], ins["goal"], ins["goal_anchor"], assets=assets, pool=goal_pool, rng=rng)
    v = (lex["verb"]["A"] + lex["verb"]["B"])[verb]
    return v.format(obj=noun, ref=ref, goal=goal)


def move_ok(g, objs, ins, plan, R, reach, rng):
    objs2 = deepcopy(objs)
    for o in objs2:
        if o["id"] == plan["anchor"]:
            o["xy"] = list(plan["xy"])
    st = g.state(objs2)
    if min_pair_gap(st, list(st)) < PRE_GAP:
        return False
    cands = [c for c in st if st[c]["role"] == "candidate"]
    r = REL.resolve(st, ins["reference"], ins["ref_anchors"], cands, R, g.table_z)
    if r["status"] != "unique" or r["holding"] != [plan["t2"]]:
        return False
    gr = REL.relation(st, ins["goal"], plan["t2"], [ins["goal_anchor"]], cands, R, g.table_z)
    if gr["holds"] or gr.get("border"):
        return False
    return place_point(st, g, ins["goal"], ins["goal_anchor"], plan["t2"], [o for o in st if o != plan["t2"]],
                       reach, rng) is not None


def build_base(world, g, reach, cell, k, gap_bin, R, seed_key, allow_flags=(), plan_move=None, tries=TRIES,
               sim_tries=12):
    rng = rng_for(seed_key)
    tgt = CANS[0]
    anchors, ga = anchor_assignment(cell, k)
    ins = instruction_parts(cell, anchors, ga)
    why, sims = {}, 0
    for _ in range(tries):
        plan = {}
        objs = propose(rng, g, cell, anchors, ga, gap_bin, R, reach, plan_move=plan_move, plan=plan)
        if objs is None:
            k2 = "propose_" + plan.get("why", "?")
            why[k2] = why.get(k2, 0) + 1
            continue
        st0 = g.state(objs)
        ok, w, _ = check_labels(st0, g, ins, tgt, gap_bin, R, reach, rng, settled=False)
        if not ok:
            why[w] = why.get(w, 0) + 1
            continue
        if raised(st0, g, ins, R) - set(allow_flags):
            why["flag_quota"] = why.get("flag_quota", 0) + 1
            continue
        if plan_move is not None and not move_ok(g, objs, ins, plan["move"], R, reach, rng):
            why["planned_move"] = why.get("planned_move", 0) + 1
            continue
        sims += 1
        st = world.place(objs)
        for o in objs:
            if o.get("on"):
                st[o["id"]]["on"] = o["on"]
        okp, w = check_physics(st, objs)
        if okp:
            ok, w, info = check_labels(st, g, ins, tgt, gap_bin, R, reach, rng, settled=True)
            if ok:
                okv, w, vis = check_visibility(world, [o["id"] for o in objs])
                if okv and not (raised(st, g, ins, R) - set(allow_flags)):
                    cands = [c for c in st if st[c]["role"] == "candidate"]
                    flags = REL.shortcut_flags(st, ins["reference"], anchors, tgt, cands, list(ANCHORS), R,
                                               g.table_z)
                    return {"cell": cell, "assignment": k, "instruction": ins, "target": tgt, "objects": objs,
                            "settled": st, "target_gap": info["target_gap"], "place_point": info["place_point"],
                            "next_to_radius": R, "flags": flags, "visibility": vis, "sim_tries": sims,
                            "planned_move": plan.get("move")}
        why["sim_" + w.split(":")[0]] = why.get("sim_" + w.split(":")[0], 0) + 1
        if sims >= sim_tries:
            break
    return {"failed": why, "sim_tries": sims}


def raised(st, g, ins, R):
    cands = [c for c in st if st[c]["role"] == "candidate"]
    f = REL.shortcut_flags(st, ins["reference"], ins["ref_anchors"], CANS[0], cands, list(ANCHORS), R, g.table_z)
    out = {k for k in ("anchor_blind", "nearest_rule") if f[k]}
    if ins["reference"] in ("nearest_to", "on_top_of"):
        out.discard("nearest_rule")
    return out


def jitter_states(world, g, reach, scene, n=5, rng=None, rad=0.01, tries=25):
    rng = rng or rng_for("jitter", scene["cell"]["cell_id"], scene["assignment"])
    out = []
    ins, gap_bin = scene["instruction"], scene.get("gap_bin_used", [0.0, 1.0])
    for _ in range(n * tries):
        objs = deepcopy(scene["objects"])
        for o in objs:
            o["xy"] = [o["xy"][0] + float(rng.uniform(-rad, rad)), o["xy"][1] + float(rng.uniform(-rad, rad))]
        st = world.place(objs)
        for o in objs:
            if o.get("on"):
                st[o["id"]]["on"] = o["on"]
        if not check_physics(st, objs)[0]:
            continue
        if check_labels(st, g, ins, scene["target"], gap_bin, scene["next_to_radius"], reach, rng, True)[0]:
            out.append({"objects": objs, "settled": st})
            if len(out) == n:
                break
    return out
