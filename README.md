# SpatialCasa: SpatialLIBERO on RoboCasa GR1

This repo builds and runs SpatialCasa-GR1, a spatial-language pick-and-place benchmark for the Fourier GR1 humanoid
(`GR1ArmsAndWaistFourierHands`) on the RoboCasa tabletop. Each scene has three identical cans and three containers,
and the instruction names one can by a reference relation and a place by a goal relation, for example "pick up the
can to the left of the pan and place it to the right of the pan".

The released data is on Hugging Face at `twanghcmut/SpatialCasa-GR1`. Its card describes the file formats, the
observation and action spec and the GR00T fine-tuning command.

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

`spatialcasa.run` handles one task at a time. It probes the 40 cells once (`tools/probe_cells.py`, saved in
`--probe`) and builds the scenes. For train it records demos and adds layouts until the task has `--target-demos`
kept demos, then writes `<out>/<task>/demos/summary.json`. For eval it fills 15 slots per task (5 cells x 3 levels).
Before building, the driver plans the 8 recipes of every slot and writes them to `scenes/plans.json`. It then draws
base scenes for each slot in a fixed seed order, builds exactly the planned recipes on each and runs the scripted
planner. A slot is done once every planned recipe is solvable on some base scene, or after `--spares` base scenes.
The default is 25; the release used 100, with `--spares-per-round` to build several at once.

`tools/export_eval.py` takes the base scene that solves the most recipes and fills the rest with the same recipes
from the spare base scenes, as in the SpatialLIBERO fallback scenes. The oracle results are in `oracle/summary.json`.

Workers split the scene list, skip finished files and start one per CPU core over the listed GPUs (about 1.4 GB of
GPU memory each), so a stopped run resumes with the same command. `--gpus` with no ids runs on CPU with OSMesa.

`spatialcasa/split.py` fixes the cells and seeds of every split. In each task 30 of the 40 cells pass the probe:
20 train cells and 10 novel cells.

| split | cells | layout seeds | items per layout | styles |
|---|---|---|---|---|
| `train` | the 20 train cells | from 0 | base + 8 sub-factors at level `train` | 0 to 5 |
| `eval_seen` | 5 train cells per task, rotated over tasks | from 1000 | 8 items of one level (below) | 6 to 11 |
| `eval_novel` | 5 novel cells per task, rotated over tasks | from 2000 | same as `eval_seen` | 6 to 11 |

Each eval cell gets one base scene per level, so the three levels never share a base scene, and each task has
5 x 3 x 8 = 120 episodes per split. The levels follow the SpatialLIBERO protocol:

| level | 8 items of a base scene |
|---|---|
| easy | each sub-factor alone at its easy value |
| medium | 4 sub-factors alone at the medium value, and 4 pairs at easy values that use each of the 8 sub-factors exactly once; the singles alternate between consecutive scenes |
| hard | 8 distinct recipes at hard values, five of 4 sub-factors and three of 5; each sub-factor in 4 or 5 of them, each dimension in at least 4, and at least 4 with a geometric sub-factor (`anchor_move`, `extra_can`, `flip`) |

`gap_near` and `gap_far` are never combined with each other, with `flip` or with `anchor_move`, and `flip` is never
combined with `anchor_move`. Sub-factors in a recipe are applied
in a fixed order (gap, anchor move, extra cans, flip, synonym, clutter, background). The hard plan of a split's 5
cells is solved as in SpatialLIBERO's `plan_hard` (sub_factor balance): an integer program over the admissible
compositions picks the tightest per-sub-factor usage window, and prefers compositions the other cells of the split
use least (`spatialcasa/recipes.py`).

A demo file `out/train/<task>/demos/<cell>_s<k>.hdf5` holds one group per kept demo with `obs/egoview`
(T x 256 x 256 x 3), `obs/state` and `actions` (T x 29), `obs/qpos`, and the instruction, target, objects and score as
attributes; `<file>.log.json` lists every attempt and why it failed. The egoview camera is mounted on the head, so the
image follows the waist. `tools/rerender_demos.py --task <Task> --needed 1100` replays stored demos, rewrites state and
frames, and moves the frames to an mp4 next to the file, which makes the folder about 5 times smaller; workers on
several machines can share one folder. `tools/export_lerobot.py` reads both forms.

## How a scene is built

The sampler places the anchors, puts the target can in a gap bin on the correct side of its anchor, and puts the
other two cans anywhere reachable and in view. Before and after a 300 step settle it checks that footprints are at
least 7 mm apart, every can is in the reach map and fully in the egoview, exactly one can satisfies the reference,
the goal is false at the start and a reachable place point exists. After the settle it also requires drift of at
most 5 mm, tilt of at most 5 degrees and at least 200 visible pixels per can.

| dimension | sub-factor | what changes |
|---|---|---|
| VSR | `background` | counter, wall and floor style; light level |
| VSR | `clutter` | unnamed objects kept off the robot's path; task objects keep 70 % of their visibility |
| SSU | `synonym` | reference, goal and verb wording from pool A (train, easy) or B (medium, hard) |
| SSU | `flip` | edited instruction (relation swap, inversion, anchor or goal change) whose answer is another can |
| OCD | `gap_near`, `gap_far` | target moved into the level's gap bin; for `between`, the anchors move apart |
| SCR | `extra_can` | one or two more identical cans that satisfy the reference with no anchor |
| SCR | `anchor_move` | the reference anchor moves so another can becomes the answer |

When a base scene has no valid gap or anchor-move item, that item draws its own base scene.

The planner grasps with one hand from the side, searches nine wrist spins for the pick and the place, drops any path
that touches another object, and uses the waist. A demo is kept when the success rule holds and a replay of the
recorded actions passes it too. Simulation steps through `World.step`, which matches `env.step` exactly and is about
1.6 times faster.

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
| `tools/probe_cells.py`, `probe_task.py`, `grasp_test.py` | diagnostics: cells, object stability, grasp success |
| `configs/levels.yaml`, `pools.yaml` | level values; style, clutter and wording pools |
