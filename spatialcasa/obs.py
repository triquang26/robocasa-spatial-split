import cv2
import numpy as np

PARTS = {"right_arm": (0, 7), "left_arm": (7, 14), "waist": (14, 17), "right_hand": (17, 23), "left_hand": (23, 29)}
RAW_PART = {"right_arm": "right", "left_arm": "left", "waist": "torso", "right_hand": "right_gripper",
            "left_hand": "left_gripper"}
IMAGE = 256
RENDER_WH = (640, 400)


def process_img(img, size=IMAGE):
    h, w, _ = img.shape
    if h != w:
        dim = max(h, w)
        y, x = (dim - h) // 2, (dim - w) // 2
        img = np.pad(img, ((y, dim - h - y), (x, dim - w - x), (0, 0)))
    if img.shape[:2] != (size, size):
        img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    return np.ascontiguousarray(img)


def ego_view(world, size=IMAGE):
    w, h = RENDER_WH
    return process_img(world.render("egoview", width=w, height=h), size)


def robot_state(env):
    from robocasa.models.robots import gather_robot_observations
    raw = gather_robot_observations(env)
    return {k: np.asarray(raw[f"robot0_{v}"], np.float64) for k, v in RAW_PART.items()}


def state29(env):
    s = robot_state(env)
    out = np.zeros(29)
    for k, (a, b) in PARTS.items():
        out[a:b] = s[k]
    return out
