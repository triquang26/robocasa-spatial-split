import hashlib
import json
import random
from collections import Counter, defaultdict

from spatialcasa import operators as OP
from spatialcasa import scene as S

ORDER = ("gap_near", "gap_far", "anchor_move", "extra_can", "flip", "synonym", "clutter", "background")
COLUMNS = ("gap_near", "gap_far", "background", "clutter", "synonym", "flip", "extra_can", "anchor_move")
EXCLUSIONS = ({"gap_near", "gap_far"}, {"flip", "anchor_move"}, {"gap_near", "flip"}, {"gap_far", "flip"},
              {"gap_near", "anchor_move"}, {"gap_far", "anchor_move"})
OWN_SCENE = ("gap_near", "gap_far", "anchor_move")
ALT_BASES = 6
SEED = 20261103
LEVELS = ("easy", "medium", "hard")
TAG = {"clean": "c", "seen": "t", "easy": "e", "medium": "m", "hard": "h", "medium_single": "s2", "hard_single": "s3"}
LEVEL_OF = {v: k for k, v in TAG.items()}
VALUE = {"clean": None, "seen": "train", "easy": "easy", "medium": "medium", "hard": "hard", "medium_single": "medium",
         "hard_single": "hard"}
SINGLE = {"medium": "medium_single", "hard": "hard_single"}
CLEAN = "c_clean"
SHORT = {"gap_near": "gapN", "gap_far": "gapF", "background": "bg", "clutter": "clut", "synonym": "syn",
         "flip": "flip", "extra_can": "xcan", "anchor_move": "amove"}


def admissible(sfs):
    return not any(ex <= set(sfs) for ex in EXCLUSIONS)


def dims(sfs):
    return sorted({OP.DIMENSION[s] for s in sfs})


def applicable(reference):
    return [s for s in COLUMNS if not (reference == "on_top_of" and s in OWN_SCENE)]


def recipe_id(level, primary=None, comps=()):
    if level == "clean":
        return CLEAN
    return f"{TAG[level]}_" + "__".join([primary, *sorted(comps, key=ORDER.index)])


def recipe_of(rid):
    tag, rest = rid.split("_", 1)
    level = LEVEL_OF[tag]
    sfs = [] if level == "clean" else rest.split("__")
    return {"id": rid, "level": level, "primary": sfs[0] if sfs else None,
            "steps": [(s, VALUE[level]) for s in sfs]}


def units(plan):
    out, singles = [], {recipe_of(r)["primary"]: r for r in plan if recipe_of(r)["level"] in SINGLE.values()}
    for r in plan:
        rc = recipe_of(r)
        if rc["level"] in SINGLE.values():
            continue
        if rc["level"] in SINGLE and rc["primary"] in singles:
            out.append((r, singles[rc["primary"]]))
        else:
            out.append((r,))
    return out


def rng(*key):
    return random.Random(int(hashlib.sha256(json.dumps([SEED, *key]).encode()).hexdigest()[:16], 16))


def companions(level, bases):
    freq, size_use, per_base, out = Counter(), Counter(), defaultdict(set), defaultdict(dict)
    r = rng("companions", level)
    slots = sorted(((b, p) for b, app in bases for p in app), key=lambda s: (s[1], s[0]))
    r.shuffle(slots)
    app_of = dict(bases)
    for b, p in slots:
        if level == "easy":
            out[b][p] = []
            continue
        allow = [c for c in COLUMNS if c != p and c in app_of[b] and admissible({p, c})]
        sizes = [1] if level == "medium" else sorted((2, 3, 4), key=lambda z: (size_use[p, z], r.random()))
        chosen = None
        for z in sizes:
            if z > len(allow):
                continue
            comp, pool = [], list(allow)
            while len(comp) < z and pool:
                r.shuffle(pool)
                pool.sort(key=lambda c: freq[p, c])
                c = next((c for c in pool if admissible({p, c, *comp})), None)
                if c is None:
                    break
                comp.append(c)
                pool.remove(c)
            if len(comp) == z and frozenset([p, *comp]) not in per_base[b]:
                chosen = comp
                break
        if chosen is None:
            alts = [c for c in allow if frozenset([p, c]) not in per_base[b]]
            r.shuffle(alts)
            alts.sort(key=lambda c: freq[p, c])
            chosen = alts[:1] if level == "medium" else (alts[:2] if len(alts) > 1 else alts)
            chosen = [c for i, c in enumerate(chosen) if admissible({p, *chosen[:i + 1]})]
        out[b][p] = sorted(chosen, key=ORDER.index)
        per_base[b].add(frozenset([p, *chosen]))
        size_use[p, len(chosen)] += 1
        for c in chosen:
            freq[p, c] += 1
    return out


def seen_plan(app):
    return [CLEAN] + [recipe_id("seen", p) for p in COLUMNS if p in app]


def level_plans(groups):
    plans = {}
    for lv in LEVELS:
        bases = [(f"{t}|{sp}|{c}", applicable(c.split("__")[0])) for t, sp, cells in groups for c in cells]
        comp = companions(lv, bases)
        for b, app in bases:
            ids = [CLEAN]
            for p in COLUMNS:
                if p not in app:
                    continue
                ids.append(recipe_id(lv, p, comp[b][p]))
                if lv in SINGLE:
                    ids.append(recipe_id(SINGLE[lv], p))
            plans[f"{b}|{lv}"] = ids
    return plans


def as_scene(prev, it):
    nxt = dict(prev)
    nxt.update({k: v for k, v in it.items() if k not in ("sub_factor", "dimension", "old_target", "preserves_target")})
    return nxt


def run_steps(ctx, base, steps, key):
    cur = base
    for sf, lv in sorted(steps, key=lambda s: ORDER.index(s[0])):
        it = OP.OPS[sf](ctx, cur, lv, S.rng_for(*key, sf, lv))
        if "failed" in it:
            return {"failed": f"{sf}:{it['failed']}", "failed_sf": sf}
        cur = as_scene(cur, it)
    return cur


def finish(base, cur, recipe, level):
    sfs = [s for s, _ in sorted(recipe["steps"], key=lambda s: ORDER.index(s[0]))]
    out = dict(cur)
    out.update({"level": recipe.get("level", level), "recipe_id": recipe["id"], "primary": recipe.get("primary"),
                "sub_factors": sfs, "op_levels": {s: lv for s, lv in recipe["steps"]}, "dimensions": dims(sfs),
                "sub_factor": "+".join(sfs) or "clean", "dimension": "+".join(dims(sfs)) or "clean",
                "old_target": base["target"], "preserves_target": cur["target"] == base["target"]})
    out.setdefault("light", 1.0)
    for k in ("jitter", "sentence", "flags"):
        out.pop(k, None)
    return out


def build(ctx, base, recipe, level, key, alt_base=None):
    cur = run_steps(ctx, base, recipe["steps"], key)
    if "failed" not in cur:
        return finish(base, cur, recipe, level)
    if cur["failed_sf"] in OWN_SCENE and alt_base is not None:
        for i in range(ALT_BASES):
            alt = alt_base(i, cur["failed_sf"], dict(recipe["steps"])[cur["failed_sf"]])
            if "failed" in alt:
                continue
            for k in ("style_id", "task"):
                alt[k] = base.get(k, alt.get(k))
            c2 = run_steps(ctx, alt, recipe["steps"], key + (i,))
            if "failed" not in c2:
                out = finish(alt, c2, recipe, level)
                out["own_base"] = {"objects": alt["objects"], "settled": alt["settled"], "key": i}
                return out
    return cur
