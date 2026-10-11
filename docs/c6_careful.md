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

## Results (2026-10-11, 3 seeds)

### Against the criteria

| Criterion | Target | C6 | Verdict |
| --- | --- | --- | --- |
| Careful-site stall | under 10% | home 26% (13-34), site 30% (24-38); C3x 100% / 99% | **Not met.** About the same as steady-cam v2 (32% / 28%) |
| Nominal speed error | within 10% of C3x (0.090) | 0.020 ± 0.000 | **Better** on 3/3 seeds, by 0.067-0.074 m/s |
| Held-out combined speed error | within 10% of C3x (0.098) | 0.041 ± 0.006 | **Better** on 3/3 seeds, by 0.053-0.064 m/s |
| Envelope | no narrower than C3x | friction 0.3 / 0.25 / 0.2 (C3x 0.15+ / 0.2 / 0.2); bumps 20 / 25+ / 20 mm (C3x 25+ on all) | **Slightly narrower** on friction (2 seeds) and bumps (2 seeds); payload, delay, shove and battery unchanged |

### What else moved

- **Yaw got worse on every seed.** Held-out mean yaw error 0.129 ± 0.018 rad/s
  vs C3x 0.085 ± 0.005 (paired +0.018 to +0.063); nominal 0.128 vs 0.052 on
  seed 1. This is the price the criteria did not anticipate. A likely cause,
  not tested: the relative yaw term has half the weight of the linear one, so
  the policy trades turning accuracy for speed accuracy.
- **Falls:** mixed (better on 1 seed, worse on 1, same on 1). Zero falls on
  the careful site.
- **Learned fast.** In the budget curve, nominal speed error falls from 0.097
  to 0.027 within the first 250 iterations and reaches 0.020 by 4499. C3x
  over the same stretch only goes from 0.098 to 0.090.
- **Readiness gap, read carefully.** Held-out minus nominal speed error is
  0.021 m/s for C6 vs 0.008 for C3x, so the gap in absolute terms grew, but
  C6 on the held-out site (0.041) is still more than twice as accurate as
  C3x at home (0.090).
- **Deployment check:** all three seeds pass (contract, numerics, cost,
  behavior).

### What it means

The stall target was missed, but the reward change did something larger than
intended: C3x's 0.09 m/s speed error at normal speed was mostly the same
leniency. The standard reward pays 90% for standing on a 0.1 m/s command and
also pays well for walking a bit too slowly at 0.3 m/s. Judging error
relative to the command fixed most of both. The cost is turning accuracy and
a little low-friction and rough-floor margin.

For the project's question, this is a commissioning finding: **the vendor's
reward definition, not the simulator's realism, set the accuracy ceiling.**
No amount of site randomization (C4, C5) touched it.

### Open

- Remaining 26-30% stall: a per-command breakdown of the careful site (which
  of the seven moves stalls) is the next diagnostic.
- A C6 v2 would raise the relative yaw weight to match the linear one and
  try a stronger stall cost; success = stall under 10% with yaw no worse
  than C3x.

## C6v2 (added 2026-10-11)

Why C6 still stalls: the vendor's training commands are drawn independently
per axis (vx ±0.4, vy ±0.3, wz ±1.0 rad/s), so a *pure* slow move almost never
comes up; the vendor's turn-in-place bucket (15% of envs) only covers
|wz| ≥ 0.4 rad/s. The careful site's 0.3 rad/s turns in place and 0.08 m/s
sidesteps are outside anything trained. Turns are about 21% of the careful
site's judged time, close to C6's remaining stall, though that is not yet
confirmed (the per-move breakdown below will show it).

**Per-move breakdown (2026-10-11, before C6v2 finished):** only half right.
Stalled share of each careful-site move, pooled over 3 seeds and both
scenarios (achieved fraction of the command in brackets):

| Move | C3x | C6 | Steady-cam v2 |
| --- | --- | --- | --- |
| vx +0.12 | 100% (0.00) | 0% (1.13) | 0% (1.27) |
| vx +0.08 | 100% (0.00) | **74%** (0.26) | 81% (0.24) |
| wz +0.30 | 100% (0.02) | 37% (0.73) | 38% (1.10) |
| vy +0.08 | 100% (0.00) | 6% (1.26) | 23% (0.54) |
| vx -0.10 | 100% (0.00) | 5% (0.92) | 17% (0.98) |
| wz -0.30 | 100% (0.02) | **67%** (0.41) | 36% (0.90) |
| vx +0.15, vy +0.05 | 95% (0.03) | 0% (1.08) | 0% (1.16) |

Weighted by judged time, C6's 26-28% splits into about half from the slowest
forward walk (0.08 m/s) and about 40% from the two turns; sidesteps and
backing up are nearly solved. C6 turns right (−0.3) worse than left. The
0.08 m/s walk may sit below the slowest gait the policy can hold (it walks
0.12 m/s at 113% of command, then mostly stops at 0.08). C6v2's bucket does
include 0.05-0.15 m/s single-axis walks, so it is a fair test of that too.

`Pretrain-C6v2-Careful-Flat-MicroDuck` = C6 plus:

| Change | C6 | C6v2 |
| --- | --- | --- |
| Careful-command bucket: share of command resamples that become one slow move on one axis | none | 20%: \|vx\| or \|vy\| 0.05-0.15 m/s, or \|wz\| 0.15-0.40 rad/s, random sign, body frame |
| `track_yaw_relative` weight | 0.5 | 1.0 (same as linear) |

Same warm start (C3 seed N), budget (1500 iterations), stall cost and world.
Two changes at once, so C6v2 vs C6 tells whether the pair works, not which
part did it.

Success: careful-site stall under 10%; held-out yaw error no worse than C3x
(0.085); speed error no worse than C6.

The evaluator now also records stall per commanded move (`per_command` in
every run); `stress_tests.py careful` adds a "Which moves stall" table.

Unattended batch (breakdown, 3 seeds, evaluation, copy to OneDrive, about 3 h):

```bash
cd ~/microduck-pretraining && git pull && bash scripts/c6v2_batch.sh
```
