import math

import numpy as np


class Frame:
    def __init__(self, base_xy, yaw, table_z):
        self.base = np.asarray(base_xy, float)[:2]
        self.yaw = float(yaw)
        self.table_z = float(table_z)
        c, s = math.cos(yaw), math.sin(yaw)
        self.R = np.array([[c, -s], [s, c]])

    def to_world(self, xy):
        return self.base + self.R @ np.asarray(xy, float)[:2]

    def to_robot(self, xy):
        return self.R.T @ (np.asarray(xy, float)[:2] - self.base)

    def yaw_to_world(self, yaw):
        return yaw + self.yaw

    def yaw_to_robot(self, yaw):
        return yaw - self.yaw

    def to_dict(self):
        return {"base_xy": self.base.tolist(), "yaw": self.yaw, "table_z": self.table_z,
                "convention": "robot frame: +x forward, +y robot's left, metres; words from the robot's view"}

    @classmethod
    def from_dict(cls, d):
        return cls(d["base_xy"], d["yaw"], d["table_z"])


def quat_wxyz_to_yaw(q):
    w, x, y, z = q
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def yaw_to_quat_wxyz(yaw):
    return np.array([math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)])
