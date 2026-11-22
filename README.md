# SpatialCasa: SpatialLIBERO on RoboCasa GR1

This repo builds and runs SpatialCasa-GR1, a spatial-language pick-and-place benchmark for the Fourier GR1 humanoid
(`GR1ArmsAndWaistFourierHands`) on the RoboCasa tabletop, simulated in MuJoCo. Each scene has three identical cans and
three containers, and the instruction names one can by a reference relation and a place by a goal relation, for
example "pick up the can to the left of the pan and place it to the right of the pan".

The released data is on Hugging Face at `twanghcmut/SpatialCasa-GR1`. Its card describes the file formats, the
observation and action spec and the GR00T fine-tuning command. The design details (sub-factors, splits, eval levels,
file formats) are in [docs/benchmark.md](docs/benchmark.md).

## How the pipeline fits together

```
configs/assets.yaml ──► task.py, assets.py      which objects a task uses (mesh, scale, grasp)
                            │
                            ▼
                        world.py                one RoboCasa env holding every object: place, settle, render
                            │
                            ▼
                        scene.py                base scene: anchors, target can, two distractor cans, checks
                            │
                            ▼
configs/levels.yaml ──► operators.py            the 8 sub-factors that edit a base scene (clutter, flip, ...)
configs/pools.yaml          │
                            ▼
                        recipes.py              which sub-factors each eval item combines
                            │
                            ▼
                        demo.py, demos.py       scripted planner (IK + reach map) records and replays demos
                            │
                            ▼
                        run.py, split.py        per-task driver over cells, seeds and GPUs
                            │
                            ▼
                        tools/export_*.py       release: eval index and scenes, LeRobot datasets
                        bench.py                evaluates a policy on the released eval scenes
```

All of these files are in `spatialcasa/` unless a path says otherwise.

## Where to change things

| to change | edit | then check with |
|---|---|---|
| an object's mesh or size | `configs/assets.yaml`: `mjcf` points at a RoboCasa asset, `scale` resizes it; the can keeps `fixed_scale` because the grasp is tuned to it | `tools/probe_task.py --task <Task>` (object stands still), `tools/grasp_test.py --task <Task>` (planner can grasp) |
| which objects a task uses | the task entry in `configs/assets.yaml`; task names are listed in `spatialcasa/task.py` | `tools/preview_levels.py --task <Task>` |
| obstacles (clutter) | object types in `configs/pools.yaml` (`clutter`), counts per level in `configs/levels.yaml`, placement in `clutter()` in `spatialcasa/operators.py` | `tools/preview_levels.py` |
| a new kind of scene edit | add an operator in `spatialcasa/operators.py` next to the existing eight and register it in `OPS` and `DIMENSION` | `tools/preview_levels.py` |
| what counts as a valid scene | `check_labels`, `check_physics`, `check_visibility` in `spatialcasa/scene.py` | `tools/probe_cells.py --task <Task>` |
| the grasp or planner | `spatialcasa/demo.py`; rerun `python -m spatialcasa.reach` afterwards | `tools/grasp_test.py` |

Clutter objects are never a can or an anchor, are kept off the robot's path, and leave the task objects at least
70 % visible. The planner drops any path that touches another object, so larger objects or more obstacles lower the
share of scenes it can solve.

## Setup

```bash
git clone https://github.com/triquang26/robocasa-spatial-split && cd robocasa-spatial-split
git clone https://github.com/robocasa/robocasa-gr1-tabletop-tasks robocasa-gr1   # tested at 4840e67
uv venv .venv --python 3.10 && source .venv/bin/activate
uv pip install -e robocasa-gr1                     # robosuite 1.5.1, mujoco 3.2.6
python robocasa-gr1/robocasa/scripts/download_kitchen_assets.py
source env.sh                                       # headless EGL; EGL_DEV=n picks the GPU
```

`env.sh` points EGL at the NVIDIA GL libraries in `../nvidia-gl-<driver>`, next to this repo (set `NVGL` to use
another folder). It leaves `CUDA_VISIBLE_DEVICES` unset because robosuite rejects an EGL device that is not listed
there.

## Evaluate a policy

```bash
huggingface-cli download twanghcmut/SpatialCasa-GR1 --repo-type dataset --include "eval/*" --local-dir SpatialCasa-GR1

python -m spatialcasa.bench --eval-dir SpatialCasa-GR1/eval --policy oracle --splits eval_seen --limit 20 --out res/oracle
python -m spatialcasa.bench --eval-dir SpatialCasa-GR1/eval --policy gr00t://localhost:5555 --out res/gr00t
python -m spatialcasa.bench --eval-dir SpatialCasa-GR1/eval --policy my_pkg.policy:MyPolicy --shard 0 8 --out res/mine
```

| flag | meaning |
|---|---|
| `--policy` | `oracle` (scripted planner, checks the harness), `gr00t://host:port` (Isaac-GR00T server, client from `gr00t.eval.robot`), or `module:Class` |
| `--splits`, `--tasks` | restrict the run, default all |
| `--limit` | at most this many episodes |
| `--shard i n` | run every n-th episode starting at i |
| `--videos k` | save mp4 rollouts of the first k episodes |
| `--out` | results folder |

A policy gets the observation dict and returns a 29-vector or a dict of `action.*` parts; an action chunk (leading
time axis) is played out step by step. An episode lasts at most 720 steps and succeeds when the goal relation holds,
the can stands within 15 degrees of upright, no other object moved more than 2 cm, and the can is released and at
rest. Results go to `results_<i>of<n>.jsonl`, so a stopped run resumes with the same command, and to `summary.json`
with success rates per split, level, dimension, sub-factor and task. A medium or hard item counts toward every
dimension and sub-factor it combines.

## Rebuild the data

```bash
source env.sh
python -m spatialcasa.reach                                   # reach map, about 12 min; after any grasp change in demo.py
python -m spatialcasa.run --split eval_seen  --tasks all --images --gpus 0 1 --out out/bench/eval_seen
python -m spatialcasa.run --split eval_novel --tasks all --images --gpus 0 1 --out out/bench/eval_novel
python -m spatialcasa.run --split train --tasks all --layouts 8 --target-demos 1000 --gpus 0 1 --out out/train
python tools/export_eval.py --src out/bench --out out/SpatialCasa-GR1/eval
python tools/export_lerobot.py --src out/train --out out/SpatialCasa-GR1/train --per-task 1000
cd tools && python merge_train.py --old <released train/> --extra ../out/train_extra --out ../out/train_merged
```

Then write the GR00T statistics for each train dataset from an Isaac-GR00T checkout:

```bash
python -c "from gr00t.data.stats import main; from gr00t.data.embodiment_tags import EmbodimentTag; \
main('out/SpatialCasa-GR1/train/gr1_spatial.PlateToPan', EmbodimentTag.ROBOCASA_GR1_TABLETOP)"
```

How the driver plans recipes, uses spare base scenes and resumes is described under "Data driver" in
[docs/benchmark.md](docs/benchmark.md).

## Files

| file | content |
|---|---|
| `spatialcasa/task.py` | the 12 GR1 tasks; `configs/assets.yaml` holds their models and scales |
| `spatialcasa/world.py` | one RoboCasa env with every object: placement, settle, state, render, visibility |
| `spatialcasa/relations.py` | spatial relations |
| `spatialcasa/scene.py`, `operators.py` | base scene sampler and the eight sub-factors |
| `spatialcasa/recipes.py` | eval level plans: which sub-factors each item combines |
| `spatialcasa/gen.py` | scene CLI: json, previews, contact sheets |
| `spatialcasa/ik.py`, `reach.py` | waist plus arm IK and the reach map `configs/reach_gr1.json` |
| `spatialcasa/demo.py`, `demos.py` | planner, recorder, scorer, replay, and the sharded demo CLI |
| `spatialcasa/run.py`, `split.py` | per-task driver and split definitions |
| `spatialcasa/obs.py` | GR00T observation: padded egoview image and the five joint-position parts |
| `spatialcasa/bench.py` | eval harness |
| `tools/export_eval.py`, `export_lerobot.py` | release export: eval index and scenes, one LeRobot v2.0 dataset per task |
| `tools/merge_train.py` | keeps a released train dataset, adds top-up demos and picks 200 clean and 100 per sub-factor per task |
| `tools/rerender_demos.py` | replay demos and store frames as mp4 |
| `tools/preview_levels.py`, `make_video.py`, `make_demo_video.py` | images and videos of scenes per level |
| `tools/probe_cells.py`, `probe_task.py`, `grasp_test.py` | diagnostics: cells, object stability, grasp success |
| `configs/levels.yaml`, `pools.yaml` | level values; style, clutter and wording pools |
