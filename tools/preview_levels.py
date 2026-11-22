import argparse
import json
import os
import textwrap
import time

from PIL import Image, ImageDraw, ImageFont

from spatialcasa import gen as G
from spatialcasa import recipes as RC
from spatialcasa import scene as S

CELL = 256
CAP = 92
LEFT = 70


def font(size):
    for p in ("/usr/share/fonts/dejavu/DejaVuSans.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        if os.path.exists(p):
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


def caption(it):
    if it is None:
        return "clean", ""
    names = [f"{RC.SHORT[s]}({it['op_levels'][s][0].upper()})" for s in it["sub_factors"]]
    return " + ".join(names), it["sentence"]


def sheet(rows, path, title):
    f1, f2, f3 = font(15), font(12), font(20)
    ncol = max(len(r[1]) for r in rows)
    W, H = LEFT + ncol * CELL, 40 + len(rows) * (CELL + CAP)
    out = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(out)
    d.text((8, 8), title, fill="black", font=f3)
    for i, (name, cols) in enumerate(rows):
        y = 40 + i * (CELL + CAP)
        d.text((6, y + CELL // 2 - 8), name, fill="black", font=f1)
        for j, (im, head, text) in enumerate(cols):
            x = LEFT + j * CELL
            if im is not None:
                out.paste(im.resize((CELL - 4, CELL - 4)), (x + 2, y + 2))
            else:
                d.rectangle([x + 2, y + 2, x + CELL - 2, y + CELL - 2], outline="gray")
            hl = textwrap.wrap(head, 38)[:2]
            for k, line in enumerate(hl):
                d.text((x + 4, y + CELL + 2 + 14 * k), line, fill=(0, 0, 160), font=f2)
            for k, line in enumerate(textwrap.wrap(text, 40)[:4]):
                d.text((x + 4, y + CELL + 6 + 14 * (len(hl) + k)), line, fill="black", font=f2)
    out.save(path, quality=92)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="PlateToPan")
    ap.add_argument("--cell", default="left_of__right_of")
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--levels", nargs="+", default=["easy", "medium", "hard"])
    ap.add_argument("--style", type=int, default=7)
    ap.add_argument("--out", default="out/preview_levels")
    a = ap.parse_args()
    ctx = G.Ctx(a.task, render=True)
    cells = {c["cell_id"]: c for c in S.cells()}
    cell = cells[a.cell]
    os.makedirs(a.out, exist_ok=True)
    rows, report = [], {}
    for li, lv in enumerate(a.levels):
        t0 = time.time()
        k = a.seed + li
        base = G.base_scene(ctx, cell, ("base", ctx["task"]["name"], a.cell, k))
        if "failed" in base:
            print(lv, "base failed", base["failed"], flush=True)
            continue
        base["style_id"] = a.style
        base["task"] = ctx["task"]["name"]
        ids = RC.level_plans([(a.task, "preview", [a.cell])])[f"{a.task}|preview|{a.cell}|{lv}"]
        recipes = [RC.recipe_of(r) for r in ids]

        def alt_base(i, sf, op_lv, k=k):
            rule = ctx["levels"]["levels"][op_lv]["anchor_move"]
            pm = (rule.get("min_disp", 0.0) if isinstance(rule, dict) else 0.0) if sf == "anchor_move" else None
            return G.base_scene(ctx, cell, ("base", a.cell, k, "alt", sf, op_lv, i), plan_move=pm)

        base_img = G.frame(ctx, base, base["style_id"])
        d = os.path.join(a.out, lv)
        os.makedirs(d, exist_ok=True)
        base_img.save(os.path.join(d, "clean.png"))
        cols = [(base_img, "clean", base["sentence"])]
        items = []
        for r in recipes:
            key = ("recipe", a.task, a.cell, k, lv)
            it = RC.build(ctx, base, r, lv, key, alt_base)
            if "failed" in it:
                print(f"  {lv:6s} {r['id']:45s} FAILED {it['failed']}", flush=True)
                cols.append((None, " + ".join(RC.SHORT[s] for s, _ in r["steps"]), "FAILED " + it["failed"]))
                items.append({"recipe": r["id"], "failed": it["failed"]})
                continue
            it["sentence"] = G.sentence_of(ctx, it)
            im = G.frame(ctx, it, it["style_id"], it["light"])
            im.save(os.path.join(d, f"{it['recipe_id']}.png"))
            with open(os.path.join(d, f"{it['recipe_id']}.json"), "w") as f:
                json.dump(G.jsonable(it), f, indent=1)
            head, text = caption(it)
            cols.append((im, head, text))
            items.append({"recipe": it["recipe_id"], "sub_factors": it["sub_factors"], "op_levels": it["op_levels"],
                          "sentence": it["sentence"], "target": it["target"], "old_target": it["old_target"],
                          "own_base": "own_base" in it})
            print(f"  {lv:6s} {it['recipe_id']:45s} {it['sentence']!r}", flush=True)
        rows.append((lv, cols))
        report[lv] = {"base_seed": k, "base_sentence": base["sentence"], "items": items,
                      "sec": round(time.time() - t0)}
        print(lv, "done", round(time.time() - t0), "s", flush=True)
    title = (f"{a.task}  cell {a.cell}  (green ring: can to pick, red ring: the answer before the change; "
             f"E/M/H = sub-factor value at easy/medium/hard)")
    sheet(rows, os.path.join(a.out, "levels.jpg"), title)
    with open(os.path.join(a.out, "levels.json"), "w") as f:
        json.dump(G.jsonable(report), f, indent=1)


if __name__ == "__main__":
    main()
