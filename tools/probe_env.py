import argparse
import json
import os

import numpy as np

import robocasa
from robocasa.utils.gym_utils.gymnasium_basic import create_env_robosuite
from robosuite.controllers import load_composite_controller_config

ROBOT = "GR1ArmsAndWaistFourierHands"


def make(env_name, seed, cams=("egoview",), size=256):
    cc = load_composite_controller_config(controller=None, robot=ROBOT)
    cc["type"] = "BASIC"
    cc["composite_controller_specific_configs"] = {}
    cc["control_delta"] = False
    env, _ = create_env_robosuite(env_name=env_name, robots=[ROBOT], controller_configs=cc,
                                  camera_names=list(cams), camera_widths=size, camera_heights=size,
                                  enable_render=True, seed=seed)
    return env


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", default="PnPCounterToPlate")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="out/probe")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    env = make(a.env, a.seed)
    obs = env.reset()
    sim = env.sim
    rep = {"env": a.env, "lang": env.get_ep_meta().get("lang")}
    base = env.robots[0].robot_model
    bid = sim.model.body_name2id(base.root_body)
    rep["robot_base_pos"] = sim.data.body_xpos[bid].tolist()
    rep["robot_base_quat"] = sim.data.body_xquat[bid].tolist()
    c = env.counter
    rep["counter"] = {"name": c.name, "pos": np.asarray(c.pos).tolist(), "size": np.asarray(c.size).tolist(), "top": float(c.pos[2] + c.size[2] / 2)}
    for k in ("get_ext_sites", "get_int_sites"):
        if hasattr(c, k):
            try:
                rep["counter"][k] = [np.asarray(p).tolist() for p in getattr(c, k)(relative=False)]
            except TypeError:
                pass
    objs = {}
    for name, obj in env.objects.items():
        b = sim.model.body_name2id(obj.root_body)
        cfg = next((x for x in env.object_cfgs if x["name"] == name), {})
        objs[name] = {"pos": sim.data.body_xpos[b].tolist(), "quat": sim.data.body_xquat[b].tolist(),
                      "bbox": np.asarray(obj.get_bbox_points()).tolist(), "hr": float(obj.horizontal_radius), "cat": cfg.get("info", {}).get("cat"),
                      "mjcf": cfg.get("info", {}).get("mjcf_path")}
    rep["objects"] = objs
    rep["cameras"] = [sim.model.camera_id2name(i) for i in range(sim.model.ncam)]
    print(json.dumps(rep, indent=1))
    with open(os.path.join(a.out, f"{a.env}_s{a.seed}.json"), "w") as f:
        json.dump(rep, f, indent=1)
    img = obs.get("egoview_image")
    if img is not None:
        from PIL import Image
        Image.fromarray(img[::-1]).save(os.path.join(a.out, f"{a.env}_s{a.seed}.png"))


if __name__ == "__main__":
    main()
