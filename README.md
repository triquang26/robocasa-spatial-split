# SpatialCasa: SpatialLIBERO on RoboCasa GR1

Spatial-language pick-and-place for the Fourier GR1 humanoid on the RoboCasa tabletop (MuJoCo). Each scene has three
identical cans and three containers. Example instruction: "pick up the can to the left of the pan and place it to
the right of the pan".

Released data: Hugging Face `twanghcmut/SpatialCasa-GR1`. Design details: [docs/benchmark.md](docs/benchmark.md).

## Setup

```bash
git clone https://github.com/triquang26/robocasa-spatial-split && cd robocasa-spatial-split
git clone https://github.com/robocasa/robocasa-gr1-tabletop-tasks robocasa-gr1   # tested at 4840e67
uv venv .venv --python 3.10 && source .venv/bin/activate
uv pip install -e robocasa-gr1                     # robosuite 1.5.1, mujoco 3.2.6
python robocasa-gr1/robocasa/scripts/download_kitchen_assets.py
source env.sh                                       # headless EGL; EGL_DEV=n picks the GPU
```

`env.sh` looks for NVIDIA GL libraries in `../nvidia-gl-<driver>`; set `NVGL` to use another folder.

## Change object shape

Edit the task entry in `configs/assets.yaml`:

- `mjcf`: mesh, a path inside the RoboCasa assets
- `scale`: size
- the can has `fixed_scale: true` because the grasp is tuned to it

Task names are in `spatialcasa/task.py`. Check:

```bash
python tools/probe_task.py --task PlateToPan      # object stands still
python tools/grasp_test.py --task PlateToPan      # planner can still grasp
python tools/preview_levels.py --task PlateToPan  # images of the scenes
```

## Add obstacles

Obstacles are the `clutter` sub-factor.

- object types: `clutter` in `configs/pools.yaml`
- count per level: `clutter` in `configs/levels.yaml`
- placement: `clutter()` in `spatialcasa/operators.py` (keeps obstacles off the robot's path, task objects stay at
  least 70 % visible)

A new kind of scene edit is a new function in `spatialcasa/operators.py`, registered in `OPS` and `DIMENSION`.

The planner drops any path that touches another object, so bigger objects or more obstacles mean fewer solvable
scenes.

## Change scene checks or the planner

- valid scene: `check_labels`, `check_physics`, `check_visibility` in `spatialcasa/scene.py`
- planner and grasp: `spatialcasa/demo.py`; rerun `python -m spatialcasa.reach` after a grasp change

## Pipeline

```
configs/assets.yaml   objects per task
world.py              RoboCasa env: place, settle, render
scene.py              base scene: anchors, target can, two other cans
operators.py          8 sub-factors that edit a scene (clutter, flip, gap, ...)
recipes.py            which sub-factors each eval item combines
demo.py, demos.py     scripted planner records demos
run.py, split.py      driver over tasks, cells, seeds, GPUs
tools/export_*.py     release: eval scenes, LeRobot datasets
bench.py              evaluate a policy
```

## Generate data

```bash
source env.sh
python -m spatialcasa.reach                                   # reach map, about 12 min
python -m spatialcasa.run --split eval_seen  --tasks all --images --gpus 0 1 --out out/bench/eval_seen
python -m spatialcasa.run --split eval_novel --tasks all --images --gpus 0 1 --out out/bench/eval_novel
python -m spatialcasa.run --split train --tasks all --layouts 8 --target-demos 1000 --gpus 0 1 --out out/train
python tools/export_eval.py --src out/bench --out out/SpatialCasa-GR1/eval
python tools/export_lerobot.py --src out/train --out out/SpatialCasa-GR1/train --per-task 1000
cd tools && python merge_train.py --old <released train/> --extra ../out/train_extra --out ../out/train_merged
```

GR00T statistics per train dataset, from an Isaac-GR00T checkout:

```bash
python -c "from gr00t.data.stats import main; from gr00t.data.embodiment_tags import EmbodimentTag; \
main('out/SpatialCasa-GR1/train/gr1_spatial.PlateToPan', EmbodimentTag.ROBOCASA_GR1_TABLETOP)"
```

## Evaluate a policy

```bash
huggingface-cli download twanghcmut/SpatialCasa-GR1 --repo-type dataset --include "eval/*" --local-dir SpatialCasa-GR1
python -m spatialcasa.bench --eval-dir SpatialCasa-GR1/eval --policy oracle --splits eval_seen --limit 20 --out res/oracle
python -m spatialcasa.bench --eval-dir SpatialCasa-GR1/eval --policy gr00t://localhost:5555 --out res/gr00t
python -m spatialcasa.bench --eval-dir SpatialCasa-GR1/eval --policy my_pkg.policy:MyPolicy --shard 0 8 --out res/mine
```

| flag | meaning |
|---|---|
| `--policy` | `oracle` (scripted planner), `gr00t://host:port`, or `module:Class` |
| `--splits`, `--tasks` | restrict the run, default all |
| `--limit` | at most this many episodes |
| `--shard i n` | every n-th episode starting at i |
| `--videos k` | mp4 of the first k episodes |
| `--out` | results folder; rerun the same command to resume |

A policy returns a 29-vector or a dict of `action.*` parts per step, or a chunk of them. Success: goal relation
holds, can within 15 degrees of upright, no other object moved more than 2 cm, can released and at rest, within 720
steps.
