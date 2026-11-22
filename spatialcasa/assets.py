import os

import robocasa

from spatialcasa import task as T
from spatialcasa.scene import load_yaml

OBJ_ROOT = os.path.join(robocasa.models.assets_root, "objects")


def clutter_model(cat):
    d = os.path.join(OBJ_ROOT, "objaverse", cat)
    m = sorted(x for x in os.listdir(d) if not x.startswith("."))[0]
    return f"objaverse/{cat}/{m}/model.xml"


def world_objects(with_clutter=True, name=None, extra=True):
    t = T.spec(name)
    P = load_yaml("pools.yaml")
    c = t["candidate"]
    ids = t["ids"] + (t["extra_ids"] if extra else ())
    objs = [dict(id=i, role="candidate", mjcf=c["mjcf"], scale=c["scale"]) for i in ids]
    for a, v in t["anchors"].items():
        objs.append(dict(id=a, role="anchor", mjcf=v["mjcf"], scale=v["scale"]))
    if with_clutter:
        for pool in ("A", "B"):
            for cat in P["clutter"][pool]:
                objs.append(dict(id=f"clutter_{cat}", role="clutter", mjcf=clutter_model(cat), scale=None))
    return objs
