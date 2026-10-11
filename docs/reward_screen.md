# Reward-weight investigation

Added 2026-10-11. Two parts: what the reward was doing while C3 got less
accurate (no training), and which of the vendor's reward weights actually
move deployment readiness (a designed screening experiment).

## Why

C3's walking accuracy is best around iteration 750 and gets worse by 3000
(`docs/stress_tests.md`), while the training reward keeps rising. C6 then
showed that changing how one reward item is scored cut nominal speed error
from 0.090 to 0.020 m/s in a few hundred iterations. So the reward
definition, not the training budget, looks like the main lever. Speed
tracking is 2 of about 17 points on offer:

| Earns points | Weight | Costs points | Weight |
| --- | --- | --- | --- |
| `air_time` | 3.0 | `foot_clearance` | -2.0 |
| `head_pose_bias` | 3.0 | `action_rate_l2` | -1.0 |
| `track_linear_velocity` | 2.0 | `dof_pos_limits` | -1.0 |
| `track_angular_velocity` | 2.0 | `self_collisions` | -1.0 |
| `upright` | 2.0 | `foot_swing_height` | -0.25 |
| `head_pose_tracking` | 2.0 | `foot_slip` | -0.1 |
| `pose` | 1.0 | `body_ang_vel` | -0.05 |
| | | `angular_momentum` | -0.02 |

(`body_pose_tracking` has weight 0 and is left out.)

## Part 1: reward terms over training (`scripts/reward_terms.py`)

Reads every `Episode_Reward/<term>` curve from the TensorBoard logs (the
term's weighted contribution per second of episode), averages over seeds,
and lines it up with the nominal speed error the budget test measured at the
same snapshots. For each term: its value at the most accurate snapshot, at
the end, the change, and its correlation with speed error from the most
accurate snapshot on. A term that rose while accuracy fell is a candidate
for what the policy traded accuracy for.

Hypotheses this can separate (from the earlier discussion):

1. **Trading accuracy for other points**: total reward and some other terms
   keep rising after ~750 while `track_linear_velocity` flattens or drops.
2. **The task changed**: `track_linear_velocity` drops during 500-2000, when
   the standing-robot share ramps from 2% to 25%, with no matching rise
   elsewhere.

Correlation across 8-10 snapshots is a pointer, not proof.

## Part 2: reward-weight screen (`scripts/reward_screen.py`)

**Question:** which of the 15 active weights change speed accuracy,
turning accuracy, falls and slow-move stalling, and in which direction?

**Design:** two-level Plackett-Burman screen (the 15 columns of a 16x16
Hadamard matrix), each weight at 0.5x or 2x its vendor value, plus 4 center
runs at the vendor weights with training seeds 1-4.

| | |
| --- | --- |
| Runs | 16 screening + 4 center = 20; centers spread through the run order (1st, 7th, 13th, 20th) to expose drift |
| Start | C3x seed 1, final checkpoint (the best standard walker) |
| Training per run | 250 iterations, 4096 envs, C3x's world, curricula at final stage (about 9 min) |
| How weights are set | `MICRODUCK_REWARD_SCALES` (JSON multipliers) read by task `Pretrain-RewardScreen-Flat-MicroDuck`; without it the task is exactly C3x |
| Scoring | final checkpoint, 1 run x 120 s on: nominal; the 8 held-out scenarios; the 2 careful-site scenarios |
| Responses (all lower is better) | nominal speed and yaw error; held-out mean speed error; held-out combined speed error; held-out mean yaw error; held-out falls/10 min; careful-site stall |
| Main effect | mean of a weight's 8 high runs minus mean of its 8 low runs |
| Noise | (a) center runs: SE = s_center x sqrt(1/8 + 1/8), t with 3 df; (b) Lenth's pseudo standard error over the 15 effects. `**` = beyond both, `*` = beyond one |

Why 250 iterations: C6's reward change had moved the policy almost all the
way within 250 iterations, and C3x's and C5's changes had also shown by
250. Why 0.5x / 2x: a factor of 4 between levels is large enough to see
through seed noise, small enough that the walker stays a walker.

**What it can and can't say.** Resolution III: each main effect is aliased
with two-weight interactions, so a big effect is a lead to confirm, not a
proof. It measures sensitivity around the vendor recipe after a short
fine-tune, not what training from scratch with those weights would give.
One seed per screening run; the center runs show how much of a difference
seed noise alone can make. The all-high run (pb01) has the vendor's
relative weights at twice the scale, so it doubles as a reward-scale check.

**Next, depending on the result:** take the 3-4 weights that matter into a
smaller follow-up with more levels and seeds (a response surface), to map
the trade-off between accuracy, falls and slow-move stalling and pick an
operating point.

## Run it

After the C6v2 batch (it waits for it):

```bash
cd ~/microduck-pretraining && git pull && bash scripts/reward_screen_batch.sh
```

Steps: budget curves for C2 and C4 (the rest cached), reward terms, the 20
screening runs, evaluation and analysis; results are copied to OneDrive
(`results/reward_terms/`, `results/reward_screen/`). Single steps:

```bash
uv run python scripts/reward_screen.py design            # the run table
uv run python scripts/reward_screen.py train --dry-run   # the 20 commands
uv run python scripts/reward_screen.py train --only ctr1 pb01
uv run python scripts/reward_screen.py evaluate
uv run python scripts/reward_screen.py analyze           # effects.md, effects.png (heatmap)
uv run python scripts/reward_terms.py --conditions c3
```

TensorBoard: `--logdir logs/rsl_rl/reward_screen`.
