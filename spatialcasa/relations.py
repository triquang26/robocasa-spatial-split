import math

import numpy as np

DIRECTIONAL = ("left_of", "right_of", "in_front_of", "behind")
AXIS = {"left_of": (1, +1), "right_of": (1, -1), "behind": (0, +1), "in_front_of": (0, -1)}
INVERSE = {"left_of": "right_of", "right_of": "left_of", "behind": "in_front_of", "in_front_of": "behind"}
REFERENCE = DIRECTIONAL + ("nearest_to", "next_to", "between", "on_top_of")
GOALS = DIRECTIONAL + ("on_top_of",)

CFG = {
    "eps_border": 0.005,
    "eps_lead": 0.005,
    "next_to_radius": 0.05,
    "between": {"t": (0.25, 0.75), "tau_perp": 0.06, "min_anchor_dist": 0.10},
    "on_inset": 0.01,
}


def centre(o):
    return np.asarray(o["xy"], float)


def gap(a, b):
    alo, ahi, blo, bhi = (np.asarray(v, float) for v in (a["box"]["lo"], a["box"]["hi"], b["box"]["lo"], b["box"]["hi"]))
    d = np.maximum(0.0, np.maximum(blo - ahi, alo - bhi))
    return float(np.hypot(*d))


def axis_gap(a, b, pred):
    k = AXIS[pred][0]
    return float(max(0.0, b["box"]["lo"][k] - a["box"]["hi"][k], a["box"]["lo"][k] - b["box"]["hi"][k]))


def directional(a, b, pred, eps=None):
    eps = CFG["eps_border"] if eps is None else eps
    k, s = AXIS[pred]
    d = centre(a) - centre(b)
    disp, orth = s * d[k], d[1 - k]
    bd = (disp - abs(orth)) / math.sqrt(2)
    return {"holds": bool(bd - eps > 0), "border": bool(abs(bd) <= eps), "border_dist": bd, "disp": disp}


def next_to(a, b, R=None, eps=None):
    R = CFG["next_to_radius"] if R is None else R
    eps = CFG["eps_border"] if eps is None else eps
    g = gap(a, b)
    return {"holds": bool(g < R), "border": bool(abs(g - R) < eps), "gap": g, "radius": R}


def between(x, a, b):
    c = CFG["between"]
    ca, cb, cx = centre(a), centre(b), centre(x)
    u = cb - ca
    L = float(np.linalg.norm(u))
    if L < c["min_anchor_dist"]:
        return {"holds": None, "border": False, "degenerate": True}
    t = float((cx - ca) @ u / L ** 2)
    e = float(abs(np.cross(u, cx - ca)) / L)
    lo, hi = c["t"]
    m = min((t - lo) * L, (hi - t) * L, c["tau_perp"] - e)
    return {"holds": bool(m > 0), "border": bool(abs(m) < CFG["eps_border"]), "t": t, "d_perp": e, "margin": m,
            "degenerate": False}


def on_top_of(x, a, table_z):
    ins = CFG["on_inset"]
    cx = centre(x)
    inside = all(a["box"]["lo"][k] + ins < cx[k] < a["box"]["hi"][k] - ins for k in (0, 1))
    base = x["z"] - x["height"] / 2
    raised = base > table_z + 0.003
    return {"holds": bool(inside and raised and x.get("tilt_deg", 0) < 30), "border": False, "inside": inside}


def relation(st, pred, x, anchors, cands=None, R=None, table_z=0.92):
    a = st[anchors[0]]
    if pred in AXIS:
        return directional(st[x], a, pred)
    if pred == "next_to":
        return next_to(st[x], a, R)
    if pred == "between":
        return between(st[x], a, st[anchors[1]])
    if pred == "on_top_of":
        return on_top_of(st[x], a, table_z)
    if pred == "nearest_to":
        g = {c: gap(st[c], a) for c in cands}
        order = sorted(g, key=lambda c: (g[c], c))
        lead = g[order[1]] - g[order[0]] if len(order) > 1 else math.inf
        ok = lead >= CFG["eps_lead"]
        return {"holds": bool(ok and x == order[0]), "border": bool(not ok and g[x] - g[order[0]] < CFG["eps_lead"]),
                "gap": g[x], "lead": lead}
    raise ValueError(pred)


def resolve(st, pred, anchors, cands, R=None, table_z=0.92):
    recs = {c: relation(st, pred, c, anchors, cands, R, table_z) for c in cands}
    if any(r.get("degenerate") for r in recs.values()):
        return {"holding": [], "border": False, "status": "degenerate", "records": recs}
    hold = [c for c, r in recs.items() if r["holds"]]
    border = any(r["border"] for r in recs.values())
    status = "border" if border else "unique" if len(hold) == 1 else "none" if not hold else "ambiguous"
    return {"holding": hold, "border": border, "status": status, "records": recs}


def target_gap(st, pred, x, anchors):
    if pred in AXIS:
        return axis_gap(st[x], st[anchors[0]], pred)
    if pred in ("nearest_to", "next_to"):
        return gap(st[x], st[anchors[0]])
    if pred == "between":
        return min(gap(st[x], st[anchors[0]]), gap(st[x], st[anchors[1]]))
    return None


def anchor_tuples(pred, anchor_ids):
    if pred == "between":
        return [(a, b) for i, a in enumerate(anchor_ids) for b in anchor_ids[i + 1:]]
    return [(a,) for a in anchor_ids]


def shortcut_flags(st, pred, anchors, target, cands, anchor_ids, R=None, table_z=0.92):
    holders, traps = set(), set()
    for tup in anchor_tuples(pred, anchor_ids):
        r = resolve(st, pred, list(tup), cands, R, table_z)
        for c in r["holding"]:
            holders.add(c)
            if tuple(tup) != tuple(anchors) and c != target:
                traps.add(c)
    near = min(cands, key=lambda c: gap(st[c], st[anchors[0]]))
    return {"anchor_blind": holders == {target}, "traps": sorted(traps), "nearest_rule": near == target}
