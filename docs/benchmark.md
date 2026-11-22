# Benchmark design

Details behind the README: how scenes, sub-factors, splits and eval levels are built, and what the
generated files hold.

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

## Splits

`spatialcasa/split.py` fixes the cells and seeds of every split. In each task 30 of the 40 cells pass the probe:
20 train cells and 10 novel cells.

| split | cells | layout seeds | items per layout | styles |
|---|---|---|---|---|
| `train` | the 20 train cells | from 0 | base + 8 sub-factors at level `train` | 0 to 5 |
| `eval_seen` | 5 train cells per task, rotated over tasks | from 1000 | 8 items of one level (below) | 6 to 11 |
| `eval_novel` | 5 novel cells per task, rotated over tasks | from 2000 | same as `eval_seen` | 6 to 11 |

## Eval levels

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

## Data driver

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

## Demo files

A demo file `out/train/<task>/demos/<cell>_s<k>.hdf5` holds one group per kept demo with `obs/egoview`
(T x 256 x 256 x 3), `obs/state` and `actions` (T x 29), `obs/qpos`, and the instruction, target, objects and score as
attributes; `<file>.log.json` lists every attempt and why it failed. The egoview camera is mounted on the head, so the
image follows the waist. `tools/rerender_demos.py --task <Task> --needed 1100` replays stored demos, rewrites state and
frames, and moves the frames to an mp4 next to the file, which makes the folder about 5 times smaller; workers on
several machines can share one folder. `tools/export_lerobot.py` reads both forms.


## Eval results

`bench.py` writes `results_<i>of<n>.jsonl` and `summary.json` with success rates per split, level, dimension,
sub-factor and task. A medium or hard item counts toward every dimension and sub-factor it combines.

## Rendering

`env.sh` leaves `CUDA_VISIBLE_DEVICES` unset because robosuite rejects an EGL device that is not listed there; pick
the GPU with `EGL_DEV`.

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
