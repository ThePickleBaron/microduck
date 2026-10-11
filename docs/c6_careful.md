# C6: careful walk (closing the slow-command gap)

Added 2026-10-10, after the stress tests (`docs/stress_tests.md`).

## Why

On the careful site (0.08-0.15 m/s and 0.3 rad/s, each from a standstill),
every policy trained with Pollen's recipe (C3, C3x, C5, the vendor policy)
stood still 97-100% of the time. The held-out site never asks for these
speeds, so the readiness report could not see it.

The cause is the reward. Speed tracking is `exp(-error^2 / 0.1)` with the
error in m/s, so standing still on a 0.1 m/s command keeps
exp(-0.01/0.1) = 90% of the reward, while walking pays every gait penalty.
The standing-robot share also ramps from 2% to 25% between iterations 500 and
2000 of C3's training, which is about where its tracking accuracy starts to
fall in the budget curve (a hypothesis, not tested here).

Steady-cam v2's stall penalty already cut slow-move stalling from ~100% to
~30% (the positive control), so the fix is known to work in part.

## What changes

`Pretrain-C6-Careful-Flat-MicroDuck` (`src/microduck_pretrain/followups.py`)
is C3x plus three reward terms. Everything else is identical to C3x: C3's
world, randomization and command ranges, warm start from C3 seed N, 1500
iterations. So **C6 vs C3x isolates the reward change.**

| Term | Weight | What it does |
| --- | --- | --- |
| `careful_stall` | -2.0 | Shortfall of the commanded motion (0..1) whenever a walk over 0.05 m/s or a turn over 0.15 rad/s is asked; free on stand commands |
| `track_lin_relative` | +1.0 | Speed tracking judged relative to the command: error / max(command, 0.1 m/s). Missing 0.08 m/s by standing scores 0.08; doing it scores 1.0 |
| `track_yaw_relative` | +0.5 | Same for turn rate (floor 0.3 rad/s) |

All three judge a 0.4 s moving average of the body velocity, not the
instantaneous one: a slow gait speeds up and slows down within every step,
and judging each instant would punish correct slow walking.

## Success criteria (decided before training)

1. Careful site: stall well below steady-cam v2's ~30%; target under 10%.
2. No price elsewhere: nominal and held-out speed error within 10% of C3x;
   operating envelope no narrower than C3x's.

If (1) holds but (2) fails, that is still a result: slow precision costs
something at normal speeds.

## Run it (desktop, about 45 min per seed)

```bash
cd ~/microduck-pretraining && git pull
uv run python scripts/train_followup.py c6 --dry-run          # parent = C3 seed N
uv run python scripts/train_followup.py c6 --seeds 1 2 3
```

TensorBoard (`--logdir logs/rsl_rl/c6_careful`):
`Episode_Reward/careful_stall` should move toward 0;
`track_lin_relative` and `track_yaw_relative` should rise; `air_time` should
stay near 1; `Episode_Termination/fell_over` should stay low.

## Evaluate it

```bash
uv run python scripts/train_followup.py --export                  # -> policies/c6_seedN.onnx
uv run python scripts/evaluate_all.py                             # nominal + held-out (skips existing)
uv run python scripts/seed_spread.py --md results/seed_spread.md  # includes c6 vs c3x, paired by seed
uv run python scripts/stress_tests.py all --conditions c6 c3x     # careful, envelope, budget (c3x cached)
```
