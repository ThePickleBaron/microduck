# C5: site fine-tune (the corrected C4), and the C3x control

Added 2026-10-09, after the first C4 runs. C1-C4 are unchanged; these are two
new conditions with their own tasks and log folders.

## What went wrong with C4

C4 (site-specific training from scratch) learned to **shuffle instead of
step**. On seed 2 at iteration 2717, compared with C2 at final-stage
curricula:

| | C2 | C4 |
| --- | --- | --- |
| Air-time reward (feet off the ground) | 1.14 | 0.01 |
| Average foot lift | 16.9 mm | 1.5 mm |
| Turn-tracking score (max 2.0) | 1.53 | 1.00 |
| Falls per episode | 0 | 0.21 |

This is the "abandon stepping" trap described in Pollen's training notes
(`third_party/microduck_rl/AGENTS.md`): when stepping is expensive early
(here gear backlash, up to 100 g payload and softer terrain contacts), the
policy settles on keeping its feet down and never leaves that basin.

Separately, **the terrain curriculum could only move robots down.** Robots
start spread over levels 0-4 of 10. mjlab's `terrain_levels_vel` demotes a
robot that covers less than half its commanded distance, but promotes one
only if it ends an episode more than half a tile (4 m) from its start. With
commands under 0.4 m/s that keep changing direction, the duck almost never
covers 4 m in 20 s, so by late training every robot sat on level 0, the
easiest, nearly flat level.

The original C4 stays in the experiment as **naive site-specific training**.
Both findings belong in the write-up.

## C5: site fine-tune

`Pretrain-C5-SiteFinetune-Rough-Backlash-MicroDuck`
(`src/microduck_pretrain/followups.py`):

- **Warm start from C3.** Seed N starts from the last checkpoint of C3 seed N,
  then fine-tunes for 1500 iterations. The policy already steps, so it never
  has to discover stepping under site conditions. This is also the better
  industrial analog: take the vendor-grade generic robot, then tune it to the
  destination site.
- **C4's site, unchanged:** rough terrain, backlash robot, 0-100 g payload,
  foot friction 0.5-1.5. Inherited curricula are pinned at their final stage
  (where the C3 policy was trained).
- **Progress-based terrain promotion** (`terrain_levels_progress`): the task
  integrates the displacement the commands asked for (in the robot's actual
  heading) and, at reset, promotes a robot that covered at least 70% of it
  along the commanded direction, demotes one below 30%, and leaves alone
  episodes that asked for under 1 m (standing robots).
- New TensorBoard curve: `Episode_Metrics/commanded_progress`, the share of
  the commanded distance actually covered.

## C3x: control

`Pretrain-C3X-Standard-Extended-Flat-MicroDuck`: the same warm start and the
same 1500 extra iterations, but in C3's own world. C5 gets 3000 + 1500
iterations of training; C3x gets exactly the same. If C5 beats C3x on the
held-out site, the gain comes from training on the site, not from training
longer.

## Run it (desktop, after the C1-C4 queue finishes)

```bash
cd ~/microduck-pretraining
git pull
uv run python scripts/train_followup.py c5 --dry-run          # shows the C3 parent for each seed
uv run python scripts/train_followup.py c5 --seeds 1 2 3      # ~1.5 h per seed (rough terrain)
uv run python scripts/train_followup.py c3x --seeds 1 2 3     # ~45 min per seed
uv run python scripts/train_followup.py --export              # -> policies/c5_seedN.onnx, c3x_seedN.onnx
uv run python scripts/evaluate_all.py                         # evaluates every policy in policies/
uv run md-report                                              # c5 and c3x appear as their own rows
```

## What to watch

In TensorBoard, runs `c5_site_finetune`:

- `Episode_Reward/air_time` should stay near C3's level (about 1.1-1.4). If it
  slides toward 0 the policy is drifting back into the shuffle; stop and tell
  Claude.
- `Curriculum/terrain_levels/mean` should climb over the run instead of
  falling to 0.
- `Episode_Metrics/commanded_progress` should rise toward 0.7 or higher.
- `Episode_Termination/fell_over` should stay low; some falls on harder
  terrain are expected.

Note: `scripts/evaluate_all.py` evaluates every ONNX file in `policies/`. If
`steadycam_seed1.onnx` is there it will be evaluated and reported too, as a
"steadycam" row; delete or move it first if you want only the conditions.
