import os

import yaml

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT = "can"
GR1_TASKS = ("CuttingboardToPan", "CuttingboardToPot", "CuttingboardToCardboardBox", "PlacematToBowl",
             "PlacematToPlate", "PlateToBowl", "PlateToPan", "PlateToPlate", "PlateToCardboardBox", "TrayToPlate",
             "TrayToPot", "TrayToCardboardBox")
_ACTIVE = {}


def registry():
    with open(os.path.join(HERE, "configs", "assets.yaml")) as f:
        return yaml.safe_load(f)


def spec(name=None):
    name = name or current()
    r = registry()
    if name not in r or name.startswith("_") or not isinstance(r[name], dict):
        raise KeyError(f"unknown task {name}; known: {sorted(r)}")
    t = dict(r[name])
    t["name"] = name
    if "source" in t:
        k = float(os.environ.get("SPATIALCASA_SCALE", r.get("gr1_scale", 1.0)))
        t["scale"] = k
        if not t["candidate"].get("fixed_scale"):
            t["candidate"] = dict(t["candidate"], scale=t["candidate"]["scale"] * k)
        t["anchors"] = {a: dict(v, scale=v["scale"] * k) for a, v in t["anchors"].items()}
    c = t["candidate"]
    key = c.get("id", c["noun"].replace(" ", "_"))
    t["ids"] = tuple(f"{key}_{i}" for i in (1, 2, 3))
    t["extra_ids"] = tuple(f"{key}_{i}" for i in (4, 5))
    t["anchor_ids"] = tuple(t["anchors"])
    t.setdefault("main", None)
    c.setdefault("grasp", "side")
    c.setdefault("nouns", {"A": [c["noun"]], "B": [c["noun"]]})
    return t


def current():
    return _ACTIVE.get("name") or os.environ.get("SPATIALCASA_TASK", DEFAULT)


def activate(name=None):
    from spatialcasa import scene as S
    t = spec(name)
    _ACTIVE["name"] = t["name"]
    os.environ["SPATIALCASA_TASK"] = t["name"]
    S.TASK = t
    S.ANCHORS = t["anchor_ids"]
    S.CANS = t["ids"]
    S.EXTRA = t["extra_ids"]
    S.MAIN = t["main"]
    return t


def reach_path(t=None):
    t = t or spec()
    g = t["candidate"]["grasp"]
    if g == "side":
        return os.path.join(HERE, "configs", "reach_gr1.json")
    return os.path.join(HERE, "configs", "reach", f"{t['name']}.json")
