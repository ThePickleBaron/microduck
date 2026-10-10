# Steady-cam walking (desktop cameraman)

A side project next to the semester experiment. It does not change C1-C4: it
adds one task, one launcher, one evaluator and a shot list.

**Goal:** the duck walks slow camera moves (dolly, truck, pan, orbit) while the
head camera stays steady enough to film a project on the desk.

## Why it needs training

The shipped walking policy was never asked to keep its camera steady. In the
CPU evaluator, walking at 0.3 m/s, its head camera sees:

| | vendor walk @ 0.3 m/s |
| --- | --- |
| Shake (angular rate, RMS) | ~69 deg/s (worst 5%: ~115 deg/s) |
| Bob (linear acceleration, RMS) | ~4.6 m/s^2 |
| Horizon tilt (RMS) | ~4.6 deg |

It also **will not start moving at cinematic speeds** in this simulator
(stalled 100% of the time on every moving shot in `sites/cinema.toml`, the
same standstill finding as in the experiment protocol). Standing still, its
camera is fine (0.7 deg/s). Baseline: `results/steadycam__vendor.json`.

## How it trains

`Steadycam-Walk-Flat-MicroDuck` (`src/microduck_pretrain/steadycam.py`):

- **Warm start from a finished walker.** Default parent since 2026-10-10:
  C3x seed N (C3 trained 1500 iterations longer), which beat C3 on tracking
  and on falls when shoved on all three seeds. Falls back to C3 if no C3x run
  exists. Walking already exists; only camera steadiness is new. 1500
  iterations, roughly 45 minutes on the 3090.
- **Three camera costs** at the `head_camera` site, ramped in from zero over
  the first 600 iterations:
  - `camera_ang_rate` - camera rotation, minus the commanded pan rate
  - `camera_lin_acc` - camera bob and jolt
  - `camera_roll` - horizon tilt
- **Head freed to stabilize.** Instantaneous head-pose tracking is lowered
  (2.0 -> 0.5); the 1 s average head-pose penalty stays at full strength. The
  head still points where it is told on average and can counter-move for a
  fraction of a second to cancel body motion, like a chicken's head.
- **Cinematic commands:** forward -0.20..0.25 m/s, sideways +-0.12 m/s,
  turning +-0.6 rad/s. Pushes stay on but softer.
- **Same 61-D observations** as every Microduck walking policy, so the ONNX is
  hot-swappable on the robot.

Weights are sized so the camera stack costs about 10% of a trained walker's
reward at the vendor policy's shake levels: enough to matter, not enough to
make standing still pay. Pollen's training notes explain why only the
*escapable* part of head motion should be priced (the head is ~38% of the
robot's mass).

## v1 result (2026-10-10): it learned to stand still

Seed 1 of v1, warm-started from C3x seed 1, on the cinema shots:

| Shot | C3x shake (deg/s) | v1 shake | C3x stalled | v1 stalled |
| --- | --- | --- | --- | --- |
| dolly_in | 72 | 1.7 | 0% | 100% |
| pan | 60 | 3.5 | 0% | 98% |
| orbit | 60 | 2.0 | 0% | 100% |
| truck | 3.5 | 0.6 | 100% | 100% |

The camera is steady because the duck stopped. The snapshot sweep
(`eval_steadycam.py --sweep steadycam`) shows it began at snapshot 250, with
the camera costs at only a third of their final weight (stalled 57-99%), and
was complete by 500. At cinematic speeds standing still keeps most of the
tracking reward (at 0.12 m/s, 87% of it) and skips every gait regularizer, so
any camera cost tipped the balance. The weights had been sized against the
vendor policy walking at 0.3 m/s, a faster gait than these shots ask for.

Separately, C3x also stalls on `truck` (0.12 m/s sideways): too slow for these
walkers to start. v2 trains with a stall penalty, which may fix that too; if
not, the shot needs a faster command.

## v2: keep walking

`Steadycam-Walk-v2-Flat-MicroDuck`, logs in `logs/rsl_rl/steadycam_walk_v2/`,
runs named `steadycam_v2_seedN`. Same as v1 plus:

- **`stall` cost, weight -2.0, full strength from iteration 0.** The shortfall
  of the commanded motion, 0..1: `1 - (v . v_cmd) / |v_cmd|^2` for a walk
  command over 0.05 m/s, the same for a turn over 0.15 rad/s. Standing still
  when asked to move costs 2 per second, about three times what the camera
  costs charge the unmodified C3x gait (~0.7 per second). Standing on a stand
  command stays free.
- **Sharper linear tracking** (std^2 0.1 -> 0.05). Angular tracking is
  unchanged: it also scores pitch and roll rates, which walking cannot avoid.
- **Camera costs ramp in over 900 iterations** instead of 600.

What to watch in TensorBoard: `Episode_Reward/stall` should stay small (near
0); if it grows while `camera_*` shrink, it is drifting toward standing again.
`air_time` should stay near the C3x level.

## Run it (desktop)

```bash
cd ~/microduck-pretraining
git pull
uv run python scripts/train_steadycam.py --dry-run     # v2; shows which C3x checkpoint it will use
uv run python scripts/train_steadycam.py               # seed 1, ~45 min
uv run python scripts/train_steadycam.py --export      # -> policies/steadycam_v2_seed1.onnx (and v1's)
uv run python scripts/eval_steadycam.py --sweep steadycam_v2   # every snapshot: did it keep walking?
```

`--seeds 1 2 3` runs several in a row; seed N starts from C3x seed N.
`--parent c3` starts from C3 instead. `--from <path>/model_XXXX.pt` picks one
specific checkpoint for every seed.

## Watch it

TensorBoard (`uv run tensorboard --logdir logs/rsl_rl --host 127.0.0.1 --port 6006`),
run `steadycam_walk`:

- `Episode_Reward/camera_*` start at 0 (curriculum), go negative as the
  weights ramp in, then shrink toward 0 as the policy learns. They must never
  be positive.
- `Episode_Reward/track_linear_velocity` and `air_time` should stay close to
  the parent C3x run. If they collapse, the camera costs are too strong.
- `Episode_Termination/fell_over` should stay near 0.

Watch it walk: `uv run play Steadycam-Walk-Flat-MicroDuck --checkpoint-file logs/rsl_rl/steadycam_walk/<run>/model_1499.pt --num-envs 16`.

## Measure it

```bash
uv run python scripts/eval_steadycam.py --policy policies/c3x_seed1.onnx --label c3x
uv run python scripts/eval_steadycam.py --policy policies/steadycam_v2_seed1.onnx --label steadycam_v2
uv run python scripts/eval_steadycam.py --compare results/steadycam__vendor.json results/steadycam__c3x.json results/steadycam__steadycam.json results/steadycam__steadycam_v2.json
```

Per shot: shake (deg/s, with the camera's own 1 s average removed so a pan
is not counted as shake), bob (m/s^2), horizon tilt (deg), falls, and the
share of time stalled. Success = lower shake and bob than its parent (C3x) on the moving
shots, with no more falls or stalls.

## Next steps

- **Roller dolly.** The roller-skate mode glides on passive wheels; a
  steady-cam variant of `Mjlab-Velocity-Flat-MicroDuck-Rollers` could give the
  smoothest dolly shots. It is a different gait (skating strides) with no
  local checkpoint to warm-start from, so it is a from-scratch run.
- **Look-at shots.** Feed head-pose commands from a target point (the
  runtime's `robot.look`) during evaluation so the orbit keeps a subject
  centered.
- **Desk-edge safety.** The 8x8 ToF tilted down can see the table edge.
- **Room mapping.** The vendored repo already ships `scene_vslam.xml` and
  `scene_apartment.xml`, simulated rooms to test a ToF-sweep mapper in.
