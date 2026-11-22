import argparse
import glob
import json
import os
import subprocess
import sys
import textwrap

import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from spatialcasa.assets import world_objects
from spatialcasa.operators import DIMENSION, SUB_FACTORS
from spatialcasa.world import World

FPS = 24
IMG = 640
PANEL = 400
HOLD_A, MOVE, HOLD_B = 14, 30, 34
F = {s: ImageFont.truetype("DejaVuSans.ttf", s) for s in (16, 18, 22)}
FB = {s: ImageFont.truetype("DejaVuSans-Bold.ttf", s) for s in (22, 30)}
DIM_NAME = {"VSR": "visual scene", "SSU": "language", "OCD": "distance", "SCR": "spatial configuration"}
WHAT = {"background": "new counter / wall / floor style and light",
        "clutter": "unnamed objects added to the counter",
        "synonym": "same task, reworded instruction",
        "flip": "edited instruction, another can is now the answer",
        "gap_near": "target moved farther from its anchor (near bin)",
        "gap_far": "target moved farther from its anchor (far bin)",
        "extra_can": "identical distractor cans that satisfy nothing",
        "anchor_move": "the reference anchor moves, another can is now the answer"}


class Worlds(dict):
    def __missing__(self, style):
        self[style] = World(world_objects(), style_id=style, size=IMG)
        return self[style]


def smooth(t):
    return t * t * (3 - 2 * t)


def pose_and_render(w, poses, light=1.0):
    for oid in w.objects:
        if oid in poses:
            xy, yaw, lift = poses[oid]
            w.write_pose(oid, xy, yaw, lift=lift)
        else:
            w.park(oid)
    w.sim.forward()
    w.set_light(light)
    img = w.render("egoview", IMG)
    w.set_light(1.0)
    return img


def support(w, objs):
    by = {o["id"]: o for o in objs}
    return {o["id"]: w.support_height(by, o) for o in objs}


def rest_poses(w, rec):
    sup = support(w, rec["objects"])
    return {o["id"]: (rec["settled"][o["id"]]["xy"], rec["settled"][o["id"]]["yaw"], 0.001 + sup[o["id"]])
            for o in rec["objects"]}


def rings(w, img, poses, rec, show_old):
    im = Image.fromarray(img)
    d = ImageDraw.Draw(im)
    marks = [(rec["target"], (0, 230, 0))]
    if show_old and rec.get("old_target") and rec["old_target"] != rec["target"]:
        marks.append((rec["old_target"], (255, 50, 50)))
    for oid, col in marks:
        if oid not in poses:
            continue
        xy = poses[oid][0]
        z = rec["settled"][oid]["z"]
        wx, wy = w.frame.to_world(xy)
        u, v = w.project([wx, wy, z], size=IMG)
        r = 34
        d.ellipse([u - r, v - r, u + r, v + r], outline=col, width=4)
    return im


def panel(head, sub, level, base_sentence, item_sentence, notes, phase_b):
    p = Image.new("RGB", (PANEL, IMG), (250, 250, 247))
    d = ImageDraw.Draw(p)
    y = 24
    d.text((20, y), head, fill=(20, 20, 20), font=FB[30])
    y += 42
    for line in textwrap.wrap(sub, 38):
        d.text((20, y), line, fill=(60, 60, 60), font=F[18])
        y += 24
    y += 8
    if level:
        d.text((20, y), f"level: {level}", fill=(60, 60, 140), font=FB[22])
        y += 40

    def para(label, text, col, y):
        d.text((20, y), label, fill=(120, 120, 120), font=F[16])
        y += 22
        for line in textwrap.wrap(text, 34):
            d.text((20, y), line, fill=col, font=F[22])
            y += 28
        return y + 14

    y = para("base instruction", base_sentence, (40, 40, 40) if not phase_b else (150, 150, 150), y)
    if phase_b and item_sentence is not None:
        y = para("item instruction", item_sentence, (10, 90, 10), y)
    for n in notes:
        for line in textwrap.wrap(n, 40):
            d.text((20, y), line, fill=(90, 60, 30), font=F[18])
            y += 24
    d.text((20, IMG - 30), "green: target   red: previous target", fill=(120, 120, 120), font=F[16])
    return p


def compose(img, pan):
    out = Image.new("RGB", (IMG + PANEL, IMG))
    out.paste(img if isinstance(img, Image.Image) else Image.fromarray(img), (0, 0))
    out.paste(pan, (IMG, 0))
    return out


def card(lines):
    im = Image.new("RGB", (IMG + PANEL, IMG), (30, 32, 38))
    d = ImageDraw.Draw(im)
    y = 200
    for i, l in enumerate(lines):
        font = FB[30] if i == 0 else F[22]
        for part in textwrap.wrap(l, 70):
            d.text((60, y), part, fill=(240, 240, 240), font=font)
            y += 44 if i == 0 else 32
        y += 10
    return im


def clip(W, base, it, sf, level):
    frames = []
    w0 = W[0]
    start_rec = dict(base)
    if "own_base" in it:
        start_rec = {"objects": it["own_base"]["objects"], "settled": it["own_base"]["settled"],
                     "target": it["old_target"], "old_target": it["old_target"], "sentence": it["sentence"]}
    A = rest_poses(w0, start_rec)
    B = rest_poses(w0, it)
    dim = DIMENSION[sf]
    head = f"{dim} / {sf}"
    sub = f"{DIM_NAME[dim]}: {WHAT[sf]}"
    notes = []
    if not it["preserves_target"]:
        notes.append(f"answer: {it['target']} (was {it['old_target']})")
    if "flip_kind" in it:
        notes.append(f"edit: {it['flip_kind']}")
    if "own_base" in it:
        notes.append("built on its own base scene")
    if it.get("target_gap") is not None and sf.startswith("gap"):
        notes.append(f"gap {100 * it['target_gap']:.1f} cm (bin {100 * it['gap_bin'][0]:g}-{100 * it['gap_bin'][1]:g} cm)")
    if sf == "background":
        notes.append(f"style {it['style_id']}, light x{it['light']:.2f}")
    if sf == "anchor_move":
        notes.append("moved " + ", ".join(f"{k} {100 * v:.0f} cm" for k, v in it["moved"].items()))

    base_sentence = start_rec.get("sentence", base["sentence"])
    imgA = rings(w0, pose_and_render(w0, A), A, start_rec, False)
    panA = panel(head, sub, level, base_sentence, None, [], False)
    frames += [compose(imgA, panA)] * HOLD_A

    panB = panel(head, sub, level, base_sentence, it["sentence"], notes, True)
    if sf == "background":
        wb = W[it["style_id"]]
        imgB = rings(wb, pose_and_render(wb, B, it["light"]), B, it, True)
        for i in range(MOVE):
            t = smooth((i + 1) / MOVE)
            frames.append(compose(Image.blend(imgA, imgB, t), panB if t > 0.5 else panA))
        frames += [compose(imgB, panB)] * HOLD_B
        return frames
    for i in range(MOVE):
        t = smooth((i + 1) / MOVE)
        P = {}
        for oid, (xy, yaw, lift) in B.items():
            if oid in A:
                xa, ya, la = A[oid]
                dyaw = (yaw - ya + np.pi) % (2 * np.pi) - np.pi
                P[oid] = ([xa[0] + t * (xy[0] - xa[0]), xa[1] + t * (xy[1] - xa[1])], ya + t * dyaw,
                          la + t * (lift - la))
            else:
                P[oid] = (xy, yaw, lift + 0.25 * (1 - ((i + 1) / MOVE) ** 2))
        img = pose_and_render(w0, P)
        frames.append(compose(rings(w0, img, P, it if t > 0.5 else start_rec, t > 0.5), panB if t > 0.5 else panA))
    w0.place(it["objects"])
    imgB = rings(w0, w0.render("egoview", IMG), B, it, True)
    frames += [compose(imgB, panB)] * HOLD_B
    return frames


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", default="out/scenes")
    ap.add_argument("--out", default="out/spatialcasa_items.mp4")
    ap.add_argument("--levels", nargs="+", default=["easy", "medium", "hard"])
    a = ap.parse_args()
    W = Worlds()
    ff = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s",
                           f"{IMG + PANEL}x{IMG}", "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-pix_fmt",
                           "yuv420p", "-crf", "20", a.out], stdin=subprocess.PIPE)
    put = lambda im, n=1: [ff.stdin.write(np.asarray(im, np.uint8).tobytes()) for _ in range(n)]
    put(card(["SpatialLIBERO on RoboCasa GR1: Counter PnP",
              "4 dimensions x 2 sub-factors, levels easy / medium / hard",
              "VSR background, clutter | SSU synonym, flip | OCD gap near, far | SCR extra can, anchor move",
              "each clip: clean base scene -> item"]), FPS * 3)
    for sdir in sorted(glob.glob(os.path.join(a.scenes, "*", "s*"))):
        base = json.load(open(os.path.join(sdir, "base.json")))
        cell = base["cell"]["cell_id"]
        put(card([f"scene {cell}", base["sentence"],
                  f"cell column: {base['cell']['column']}"]), int(FPS * 2.5))
        A = rest_poses(W[0], base)
        put(compose(rings(W[0], pose_and_render(W[0], A), A, base, False),
                    panel("clean base scene", "three identical cans, three named anchors", None, base["sentence"],
                          None, [], False)), FPS * 2)
        for sf in SUB_FACTORS:
            for lv in a.levels:
                p = os.path.join(sdir, lv, f"{sf}.json")
                if not os.path.exists(p):
                    continue
                for fr in clip(W, base, json.load(open(p)), sf, lv):
                    put(fr)
                print(cell, sf, lv, flush=True)
    ff.stdin.close()
    ff.wait()
    print("wrote", a.out)


if __name__ == "__main__":
    main()
