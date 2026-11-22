import argparse
import glob
import json
import os
import subprocess
import sys

import h5py
import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from make_video import DIM_NAME, IMG, PANEL, WHAT, Worlds, card, compose, panel
from spatialcasa import demo
from spatialcasa.operators import DIMENSION, SUB_FACTORS

FPS = 20
STRIDE = 2
HOLD = 20


def notes_for(it, sf):
    out = []
    if not it.get("preserves_target", True):
        out.append(f"answer: {it['target']} (was {it['old_target']})")
    if "flip_kind" in it:
        out.append(f"edit: {it['flip_kind']}")
    if it.get("target_gap") is not None and sf.startswith("gap"):
        out.append(f"gap {100 * it['target_gap']:.1f} cm")
    if sf == "background":
        out.append(f"style {it['style_id']}, light x{it['light']:.2f}")
    if sf == "anchor_move":
        out.append("moved " + ", ".join(f"{k} {100 * v:.0f} cm" for k, v in it["moved"].items()))
    return out


def ring(w, img, rec):
    im = Image.fromarray(img)
    d = ImageDraw.Draw(im)
    st = w.state([rec["target"]])[rec["target"]]
    u, v = w.project([*w.frame.to_world(st["xy"]), st["z"]], size=IMG)
    d.ellipse([u - 34, v - 34, u + 34, v + 34], outline=(0, 230, 0), width=4)
    return im


def status(score, side, steps):
    return [f"hand: {side}, {steps} steps ({steps / demo.HZ:.1f} s)",
            f"goal holds: {score['goal']}, moved others: {100 * score['disturbance']:.1f} cm"]


def clip(W, rec, actions, head, sub, level, base_sentence, notes, score):
    w = W[rec.get("style_id", 0)]
    w.place(rec["objects"])
    w.set_light(rec.get("light", 1.0))
    extra = status(score, score["side"], len(actions))
    pan = panel(head, sub, level, base_sentence, rec["sentence"], notes + extra, True)
    frames = []
    for t, a in enumerate(actions):
        if t % STRIDE == 0:
            frames.append(compose(ring(w, w.render("egoview", IMG), rec), pan))
        w.step(a)
    frames += [compose(ring(w, w.render("egoview", IMG), rec), pan)] * HOLD
    w.set_light(1.0)
    return frames


def failed_card(head, level, sentence, reason):
    return card([f"{head} ({level}): no demo kept", sentence, f"reason: {reason}"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", default="out/scenes")
    ap.add_argument("--demos", default="out/demos")
    ap.add_argument("--out", default="out/spatialcasa_demo_items.mp4")
    ap.add_argument("--levels", nargs="+", default=["easy", "medium", "hard"])
    a = ap.parse_args()
    log = {(r["cell"], r["scene"], r["item"]): r for r in json.load(open(os.path.join(a.demos, "log.json")))}
    W = Worlds()
    ff = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s",
                           f"{IMG + PANEL}x{IMG}", "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-pix_fmt",
                           "yuv420p", "-crf", "22", a.out], stdin=subprocess.PIPE)

    def put(im, n=1):
        for _ in range(n):
            ff.stdin.write(np.asarray(im, np.uint8).tobytes())

    put(card(["SpatialLIBERO on RoboCasa GR1: robot demos per item",
              "each clip: the scripted GR1 demo for one item, replayed from the recorded actions",
              "VSR background, clutter | SSU synonym, flip | OCD gap near, far | SCR extra can, anchor move",
              "green ring: the can the instruction asks for"]), FPS * 3)
    for sdir in sorted(glob.glob(os.path.join(a.scenes, "*", "s*"))):
        cell, scene = sdir.split(os.sep)[-2:]
        base = json.load(open(os.path.join(sdir, "base.json")))
        h5 = h5py.File(os.path.join(a.demos, f"{cell}_{scene}.hdf5"), "r")
        put(card([f"scene {cell}", base["sentence"]]), int(FPS * 2))
        items = [("base", base, "clean base scene", "three identical cans, three named anchors", None)]
        for sf in SUB_FACTORS:
            for lv in a.levels:
                p = os.path.join(sdir, lv, f"{sf}.json")
                if os.path.exists(p):
                    items.append((f"{lv}/{sf}", json.load(open(p)), f"{DIMENSION[sf]} / {sf}",
                                  f"{DIM_NAME[DIMENSION[sf]]}: {WHAT[sf]}", lv))
        for name, rec, head, sub, lv in items:
            r = log[(cell, scene, name)]
            if f"data/{name}" not in h5:
                put(failed_card(head, lv or "base", rec["sentence"], r.get("error") or
                                f"goal {r.get('goal')}, tilt {r.get('tilt', 0):.0f} deg"), FPS * 2)
                print(cell, name, "no demo", flush=True)
                continue
            g = h5[f"data/{name}"]
            sf = name.split("/")[-1]
            notes = notes_for(rec, sf) if lv else []
            for fr in clip(W, rec, g["actions"][:], head, sub, lv, base["sentence"], notes, r):
                put(fr)
            print(cell, name, flush=True)
    ff.stdin.close()
    ff.wait()
    print("wrote", a.out)


if __name__ == "__main__":
    main()
