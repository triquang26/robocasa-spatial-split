import argparse
import glob
import json
import math
import os
import shutil
import subprocess
import sys
import time
from collections import Counter, defaultdict

from spatialcasa import recipes as RC
from spatialcasa import split as SP
from spatialcasa import task as T

HERE = T.HERE
MB_PER_DEMO = 6.5


def sh(cmd, env, log):
    with open(log, "a") as f:
        return subprocess.Popen(cmd, env=env, stdout=f, stderr=subprocess.STDOUT, cwd=HERE)


def worker_env(gpu, a):
    env = dict(os.environ, PYTHONPATH=HERE, PYTHONUNBUFFERED="1", OMP_NUM_THREADS=str(a.threads))
    if gpu is None:
        env.update(MUJOCO_GL="osmesa", PYOPENGL_PLATFORM="osmesa", CUDA_VISIBLE_DEVICES="", SPATIALCASA_IK_DEVICE="cpu")
    else:
        env.update(MUJOCO_GL="egl", PYOPENGL_PLATFORM="egl", MUJOCO_EGL_DEVICE_ID=str(gpu), EGL_DEV=str(gpu),
                   SPATIALCASA_IK_DEVICE=f"cuda:{a.cuda_map.get(gpu, gpu)}")
        env.pop("CUDA_VISIBLE_DEVICES", None)
    if a.ik_compile:
        env["SPATIALCASA_IK_COMPILE"] = a.ik_compile
    return env


def slots(a):
    if not a.gpus:
        return [None] * a.cpu_workers
    per = a.workers_per_gpu or max(1, (os.cpu_count() or 1) // len(a.gpus))
    return [g for _ in range(per) for g in a.gpus]


def run_all(cmds, a, logdir, name):
    os.makedirs(logdir, exist_ok=True)
    procs = []
    for i, (gpu, cmd) in enumerate(cmds):
        procs.append(sh(cmd, worker_env(gpu, a), os.path.join(logdir, f"{name}_{i}.log")))
    codes = [p.wait() for p in procs]
    bad = [i for i, c in enumerate(codes) if c]
    if bad:
        print(f"  {name}: workers {bad} exited non-zero; see {logdir}/{name}_<i>.log", flush=True)
    return not bad


def need_gb(a, tasks):
    cells = 20 if a.split == "train" else a.n_cells
    per_task = cells * a.layouts * (1 + len(a.sub_factors) * len(a.levels))
    return 0 if a.eval else per_task * len(tasks) * MB_PER_DEMO / 1024


def summarize(task, droot, sroot):
    logs = []
    for p in sorted(glob.glob(os.path.join(droot, "*.hdf5.log.json"))):
        logs += json.load(open(p))
    per_cell = defaultdict(lambda: [0, 0])
    why = Counter()
    secs = []
    for r in logs:
        per_cell[r["cell"]][1] += 1
        per_cell[r["cell"]][0] += int(r["success"])
        if "sec" in r:
            secs.append(r["sec"])
        if not r["success"]:
            for side, e in (r.get("error") or {}).items():
                why[f"{side}:{e}"] += 1
    base_fail = Counter()
    for p in glob.glob(os.path.join(sroot, "*", "s*", "done.json")):
        d = json.load(open(p))
        if d.get("base") == "failed":
            base_fail[p.split(os.sep)[-3]] += 1
    out = {"task": task, "kept": sum(v[0] for v in per_cell.values()), "tried": len(logs),
           "per_cell": {c: {"kept": v[0], "tried": v[1]} for c, v in sorted(per_cell.items())},
           "top_failures": why.most_common(10), "sec_per_item": round(sum(secs) / max(len(secs), 1), 1),
           "base_scene_failures": dict(base_fail)}
    with open(os.path.join(droot, "summary.json"), "w") as f:
        json.dump(out, f, indent=1)
    with open(os.path.join(droot, "log.json"), "w") as f:
        json.dump(logs, f, default=float)
    return out


def do_task(task, a):
    t0 = time.time()
    root = os.path.join(a.out, task)
    sroot, droot, logdir = os.path.join(root, "scenes"), os.path.join(root, "demos"), os.path.join(root, "logs")
    os.makedirs(logdir, exist_ok=True)
    free = shutil.disk_usage(a.out).free / 2 ** 30
    print(f"[{task}] free {free:.0f} GB", flush=True)
    if free < need_gb(a, [task]) + a.min_free_gb:
        print(f"[{task}] STOP: need ~{need_gb(a, [task]):.0f} GB", flush=True)
        return None
    if a.split in ("eval_seen", "eval_novel"):
        return do_eval_split(task, a, root, sroot, logdir, t0)
    probe = os.path.join(a.probe, f"{task}_cells.json")
    if not os.path.exists(probe):
        print(f"[{task}] probe cells", flush=True)
        cmd = [sys.executable, "tools/probe_cells.py", "--task", task, "--out", a.probe]
        run_all([(slots(a)[0], cmd)], a, logdir, "probe")
    n_ok = json.load(open(probe))["ok"]
    if n_ok < a.n_cells:
        print(f"[{task}] only {n_ok} cells pass the probe (< {a.n_cells}); using all {n_ok}", flush=True)
    styles = "B" if a.eval else "A"
    S = slots(a)
    cells = SP.split_for(task, a.probe)["train_cells"] if a.split == "train" else None
    pick = ["--cells", *cells] if cells else []
    n_cells = len(cells) if cells else a.n_cells
    layouts = a.layouts
    while True:
        gen = [(g, [sys.executable, "-m", "spatialcasa.gen", "--task", task, *pick, "--n-cells", str(a.n_cells),
                    "--probe", a.probe, "--layouts", str(layouts), "--start",
                    str(a.start if a.start is not None else (1000 if a.eval else 0)),
                    "--styles", styles, "--levels", *a.levels,
                    "--sub-factors", *a.sub_factors, "--out", sroot, "--resume", "--shard", str(i), str(len(S))]
                + (["--images"] if a.images else []))
               for i, g in enumerate(S)]
        print(f"[{task}] scenes: {layouts} layouts, {len(S)} workers", flush=True)
        run_all(gen, a, logdir, "gen")
        print(f"[{task}] scenes done {time.time() - t0:.0f}s", flush=True)
        if a.eval:
            return {"task": task, "scenes_sec": round(time.time() - t0)}
        t1 = time.time()
        dem = [(g, [sys.executable, "-m", "spatialcasa.demos", "--task", task, "--scenes", sroot, "--out", droot,
                    "--levels", *a.levels, "--sub-factors", *a.sub_factors, "--resume", "--shard", str(i),
                    str(len(S)), "--min-free-gb", str(a.min_free_gb)])
               for i, g in enumerate(S)]
        print(f"[{task}] demos: {len(S)} workers", flush=True)
        run_all(dem, a, logdir, "demos")
        s = summarize(task, droot, sroot)
        s["layouts"] = layouts
        s["demos_sec"] = round(time.time() - t1)
        s["total_sec"] = round(time.time() - t0)
        with open(os.path.join(droot, "summary.json"), "w") as f:
            json.dump(s, f, indent=1)
        print(f"[{task}] kept {s['kept']}/{s['tried']}  layouts {layouts}  {s['sec_per_item']} s/item  "
              f"{s['demos_sec']} s wall", flush=True)
        for c, v in s["per_cell"].items():
            print(f"    {c:28s} {v['kept']}/{v['tried']}", flush=True)
        print(f"    top failures {s['top_failures'][:3]}", flush=True)
        short = a.target_demos - s["kept"]
        if short <= 0 or layouts >= a.max_layouts:
            return s
        per_layout = max(s["kept"] / max(layouts, 1), n_cells)
        layouts = min(layouts + math.ceil(1.1 * short / per_layout), a.max_layouts)


def oracle_logs(oroot):
    out = defaultdict(dict)
    for p in glob.glob(os.path.join(oroot, "*.oracle.json")):
        rows = read_json(p)
        if rows is None:
            os.remove(p)
            continue
        for r in rows:
            out[r["cell"], r["scene"]][r["item"]] = r
    return out


def read_json(p):
    try:
        with open(p) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def slot_status(task, a, sroot, oroot):
    orc = oracle_logs(oroot)
    res = []
    for sl in eval_slots(task, a):
        tried, plan, cover = [], a.plans[f"{sl['cell']}|{sl['level']}"], {}
        for r in range(a.spares):
            got = seed_of(task, a, sl, r)
            if got is None:
                break
            cc, k = got
            d = os.path.join(sroot, cc, f"s{k}")
            if not os.path.exists(os.path.join(d, "done.json")):
                break
            done = read_json(os.path.join(d, "done.json"))
            if done is None:
                shutil.rmtree(d, ignore_errors=True)
                break
            o = orc.get((cc, f"s{k}"))
            items = done.get("items") or {}
            solved = {}
            for rid in plan:
                if items.get(rid) == "ok" and o and o.get(f"{sl['level']}/{rid}", {}).get("success"):
                    solved[rid] = rid
            for u in RC.units(plan):
                if all(x in solved for x in u):
                    cover.setdefault("+".join(u), k)
            tried.append({"seed": k, "cell": cc, "base": done["base"], "oracle": o is not None, "solved": solved,
                          "solvable": len(solved), "planned": len(plan),
                          "built": sum(items.get(rid) == "ok" for rid in plan)})
        full = bool(plan) and all("+".join(u) in cover for u in RC.units(plan))
        res.append(dict(sl, plan=plan, tried=tried, cover=cover, complete=full))
    return res


def eval_summary(task, a, status, oroot):
    by = defaultdict(lambda: [0, 0])
    for sl in status:
        for t in sl["tried"]:
            by[f"level:{sl['level']}"][0] += t["solvable"]
            by[f"level:{sl['level']}"][1] += t["planned"]
        by[f"slot_items:{sl['level']}"][0] += sum(len(u.split("+")) for u in sl["cover"])
        by[f"slot_items:{sl['level']}"][1] += len(sl["plan"])
    out = {"task": task, "split": a.split, "slots": len(status), "complete": sum(sl["complete"] for sl in status),
           "bases_tried": sum(len(sl["tried"]) for sl in status),
           "items_solvable_over_planned": {k: {"solvable": v[0], "planned": v[1]} for k, v in sorted(by.items())},
           "slots_detail": status}
    os.makedirs(oroot, exist_ok=True)
    with open(os.path.join(oroot, "summary.json"), "w") as f:
        json.dump(out, f, indent=1)
    return out


def eval_slots(task, a):
    if not a.seen:
        return SP.level_slots(task, a.probe, a.split, a.per_task)
    cells = [s["cell"] for s in SP.level_slots(task, a.probe, "eval_seen") if s["level"] == SP.EVAL_LEVELS[0]]
    return [{"level": "seen", "index": j, "cell": c} for j, c in enumerate(cells)]


def seen_candidates(task, a, sl):
    rows = [x.split("|") for x in a.seen_rows if x.startswith(f"{task}|")]
    own = sorted((c, int(k)) for _, c, k, _ in rows if c == sl["cell"])
    taken = {s["cell"] for s in eval_slots(task, a)}
    others = sorted({c for _, c, _, _ in rows if c not in taken})
    mine = set(others[sl["index"]::len(taken)])
    return own + sorted(((c, int(k)) for _, c, k, _ in rows if c in mine), key=lambda x: (x[1], x[0]))


def seed_of(task, a, sl, r):
    if sl["level"] != "seen":
        return sl["cell"], SP.slot_seed(task, a.split, sl["level"], r)
    cands = seen_candidates(task, a, sl)
    return cands[r] if r < len(cands) else None


def plans_for(task, split, probe):
    groups = [(t, sp, [s["cell"] for s in SP.level_slots(t, probe, sp) if s["level"] == SP.EVAL_LEVELS[0]])
              for t in T.GR1_TASKS for sp in ("eval_seen", "eval_novel")]
    pre = f"{task}|{split}|"
    return {k[len(pre):]: v for k, v in RC.level_plans(groups).items() if k.startswith(pre)}


def do_eval_split(task, a, root, sroot, logdir, t0):
    oroot = os.path.join(root, "oracle")
    a.plans = plans_for(task, a.split, a.probe) if not a.seen else {
        f"{c}|seen": RC.seen_plan(RC.applicable(c.split("__")[0]))
        for c in {x.split("|")[1] for x in a.seen_rows if x.startswith(f"{task}|")}}
    os.makedirs(sroot, exist_ok=True)
    plans_path = os.path.join(sroot, "plans.json")
    with open(plans_path, "w") as f:
        json.dump(a.plans, f, indent=1)
    rnd, retried = 0, set()
    while True:
        status = slot_status(task, a, sroot, oroot)
        jobs, pending, needs = [], [], {}
        for sl in status:
            if sl["complete"]:
                continue
            wait = [os.path.join(sroot, t.get("cell", sl["cell"]), f"s{t['seed']}") for t in sl["tried"]
                    if t["base"] == "ok" and not t["oracle"]]
            wait = [d for d in wait if d not in retried]
            retried.update(wait)
            if wait:
                pending += wait
                continue
            todo = [x for u in RC.units(sl["plan"]) if "+".join(u) not in sl["cover"] for x in u]
            n_new = 1 if not sl["tried"] else a.spares_per_round
            for r in range(len(sl["tried"]), min(len(sl["tried"]) + n_new, a.spares)):
                got = seed_of(task, a, sl, r)
                if got is None:
                    break
                cc, k = got
                jobs.append(f"{cc}:{k}:{sl['level']}:{sl['index']}")
                if sl["tried"]:
                    needs[f"{cc}|s{k}"] = todo
        s = eval_summary(task, a, status, oroot)
        print(f"[{task}] {a.split} round {rnd}: complete {s['complete']}/{s['slots']}  bases {s['bases_tried']}  "
              f"{round(time.time() - t0)} s", flush=True)
        if not jobs and not pending:
            s["total_sec"] = round(time.time() - t0)
            with open(os.path.join(oroot, "summary.json"), "w") as f:
                json.dump(s, f, indent=1)
            return s
        needs_path = os.path.join(sroot, f"needs_r{rnd}.json")
        with open(needs_path, "w") as f:
            json.dump(needs, f)
        S = slots(a)[:max(1, len(jobs))]
        gen = [(g, [sys.executable, "-m", "spatialcasa.gen", "--task", task, "--level-jobs", *jobs, "--plans", plans_path, "--needs", needs_path, "--styles", "B",
                    *(["--seen-layouts", a.seen_layouts] if a.seen else []),
                    "--out", sroot, "--resume", "--shard", str(i), str(len(S))] + (["--images"] if a.images else []))
               for i, g in enumerate(S)]
        if jobs:
            run_all(gen, a, logdir, f"gen_r{rnd}")
        dirs = [os.path.join(sroot, j.split(":")[0], f"s{j.split(':')[1]}") for j in jobs]
        dirs = [d for d in dirs if os.path.exists(os.path.join(d, "base.json"))] + pending
        S = slots(a)[:max(1, len(dirs))]
        dem = [(g, [sys.executable, "-m", "spatialcasa.demos", "--task", task, "--scenes", sroot, "--out", oroot,
                    "--levels", *(["seen"] if a.seen else SP.EVAL_LEVELS), "--recipes", "--resume", "--oracle", "--dirs", *dirs[i::len(S)]])
               for i, g in enumerate(S) if dirs[i::len(S)]]
        run_all(dem, a, logdir, f"oracle_r{rnd}")
        rnd += 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", nargs="+", default=["PlateToBowl"])
    ap.add_argument("--layouts", type=int, default=10)
    ap.add_argument("--n-cells", type=int, default=8)
    ap.add_argument("--levels", nargs="+", default=None)
    ap.add_argument("--sub-factors", nargs="+", default=["background", "clutter", "synonym", "flip", "gap_near",
                                                          "gap_far", "extra_can", "anchor_move"])
    ap.add_argument("--eval", action="store_true")
    ap.add_argument("--split", choices=["train", "eval_seen", "eval_novel"], default=None)
    ap.add_argument("--per-task", type=int, default=SP.EVAL_PER_TASK)
    ap.add_argument("--spares", type=int, default=SP.SPARES)
    ap.add_argument("--spares-per-round", type=int, default=1)
    ap.add_argument("--seen", action="store_true")
    ap.add_argument("--seen-layouts", default=None)
    ap.add_argument("--target-demos", type=int, default=0)
    ap.add_argument("--max-layouts", type=int, default=60)
    ap.add_argument("--start", type=int, default=None)
    ap.add_argument("--images", action="store_true")
    ap.add_argument("--gpus", type=int, nargs="*", default=[])
    ap.add_argument("--cuda-map", default="")
    ap.add_argument("--workers-per-gpu", type=int, default=None)
    ap.add_argument("--cpu-workers", type=int, default=os.cpu_count() or 1)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--ik-compile", default="none")
    ap.add_argument("--probe", default="out/probe/v2")
    ap.add_argument("--min-free-gb", type=float, default=20.0)
    ap.add_argument("--out", default="out/data")
    a = ap.parse_args()
    a.tasks = list(T.GR1_TASKS) if a.tasks == ["all"] else a.tasks
    a.spares = min(a.spares, SP.SLOT_STRIDE + SP.OVERFLOW_STRIDE)
    a.seen_rows = json.load(open(a.seen_layouts)) if a.seen else []
    if a.seen_layouts:
        a.seen_layouts = os.path.abspath(a.seen_layouts)
    a.eval = a.eval or (a.split or "").startswith("eval")
    a.levels = a.levels or (SP.EVAL_LEVELS if a.eval else ["train"])
    a.cuda_map = {int(k): int(v) for k, v in (x.split(":") for x in a.cuda_map.split(",") if x)}
    a.out = os.path.abspath(a.out)
    a.probe = os.path.abspath(a.probe)
    os.makedirs(a.out, exist_ok=True)
    print(f"tasks {a.tasks} layouts {a.layouts} cells {a.n_cells} levels {a.levels} eval {a.eval} "
          f"workers {len(slots(a))} est disk {need_gb(a, a.tasks):.0f} GB", flush=True)
    res = {}
    for task in a.tasks:
        r = do_task(task, a)
        if r is None:
            break
        res[task] = r
    with open(os.path.join(a.out, f"{a.split or ('eval' if a.eval else 'train')}_summary.json"), "w") as f:
        json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
