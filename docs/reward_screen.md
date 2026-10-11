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

## Results (2026-10-11)

### Part 1: what the reward pays for

**C4's learning curve (the "not enough iterations?" question).** Nominal speed
error is 0.168 m/s at iteration 0, 0.213 (= standing still) from iteration
250, and stays at 0.213-0.214 at every snapshot through 2999 on all three
seeds. Its training reward meanwhile climbs from 1.4 to 114 (episodes last
the full 20 s; the logged fall terminations drop from 97.6 to 0.13). C4 settled on standing within
250 iterations and got better at standing for the next 2750; more
iterations would not have changed that. `air_time` (paid only for steps
lasting 0.125-0.3 s) is 0.012/s for C4 versus ~1.0/s for C3: its feet
essentially never leave the ground.

**Why standing pays.** `track_linear_velocity` is exp(-error^2 / 0.1), and the
error includes vertical body velocity. Under the final command mix (25%
standing envs, 15% turn-in-place, the rest uniform up to 0.4 m/s) a robot
that never moves would earn about 0.68 of the maximum, 1.36 per second of
the 2.0 available (Monte Carlo over the command distribution; ignores falls
and randomization). The logged values at iteration 2999: C3 (walks) 1.29,
C4 (stands) 1.28. **The speed-tracking term pays a walker no more than a
statue.** `track_angular_velocity` also penalizes roll and pitch rate
(wobble), so walking costs angular reward: a statue would earn about 1.35/s;
C3 earns 0.73, C4 1.07. In this reward, walking is paid for mainly by
`air_time` (weight 3.0).

**C3's drift after iteration 750** (speed error 0.063 -> 0.099 m/s):

| Term | At 750 | At 2999 | Change | r with speed error, 750-2999 |
| --- | --- | --- | --- | --- |
| track_angular_velocity | +0.246 | +0.727 | +0.481 | +0.89 |
| air_time | +1.241 | +0.980 | -0.261 | -0.94 |
| head_pose_bias | -0.047 | -0.279 | -0.232 | -0.96 |
| track_linear_velocity | +1.183 | +1.287 | +0.104 | +0.91 |
| total (Train/mean_reward) | 111.0 | 111.9 | +0.9 | |

Total reward is flat: the policy re-allocated, not improved. It earned
three times more angular reward (less wobble, better turning) and less
`air_time` (less stepping), and walked less accurately. That is hypothesis 1
(trading accuracy for other points) with a specific trade: speed for
steadiness. The standing-share ramp (hypothesis 2) can contribute to the
small `track_linear_velocity` rise, but speed error kept rising after the
ramp ended at 2000 (0.088 -> 0.099), so it is not the whole story.

### Part 2: the weight screen

Effect of raising each weight from 0.5x to 2x (negative = better).
Center-run noise thresholds in the last row; `**` beyond both thresholds.
Heatmap: `results/reward_screen/effects.png`.

| Weight | Nominal speed | Held-out speed | Held-out yaw | Falls/10 min | Careful stall (%) |
| --- | --- | --- | --- | --- | --- |
| track_linear_velocity | **-0.109** | **-0.104** | **+0.068** | +0.4 | -4.7 |
| track_angular_velocity | +0.045 | +0.041 | **-0.091** | -0.5 | +4.9 |
| air_time | -0.025 | -0.021 | -0.019 | +0.2 | **-17.7** |
| pose | +0.045 | +0.043 | +0.011 | -0.1 | +1.4 |
| self_collisions | -0.028 | -0.025 | -0.008 | +0.2 | +1.4 |
| dof_pos_limits | +0.025 | +0.021 | -0.008 | +0.2 | +1.8 |
| action_rate_l2 | -0.006 | -0.002 | -0.032 | +0.1 | -4.7 |
| *noise threshold (center runs, 95%)* | 0.011 | 0.009 | 0.010 | 0.5 | 4.6 |

1. **Speed and turning are a seesaw.** The two tracking weights dominate
   every accuracy response, in opposite directions. Runs with the linear
   weight high and the angular weight low walk accurately (held-out speed
   0.03-0.06 m/s) but turn badly (yaw 0.21-0.24 rad/s); the reverse turns well
   (0.05-0.09) and barely walks (0.18-0.21). C3's drift and C6's turning regression are
   the same trade, made by training rather than by us.
2. **The vendor weights sit near a cliff.** With `track_linear_velocity`
   at half weight, all 8 runs walked far worse within 250 iterations
   (nominal speed error 0.14-0.21; standing still = 0.213) and two had
   stopped completely. One run with both tracking weights doubled (pb09)
   also stopped, so other weights can tip it too.
3. **`air_time` is the stepping incentive.** Doubling it cut slow-move
   stalling by 18 points, the only stall effect beyond both thresholds,
   and helped speed accuracy a little. Consistent with Part 1: stepping,
   not speed tracking, is what pays a walker.
4. **Falls barely move** (all runs at or under 1.25 per 10 min): within this
   range the weights trade accuracy, not robustness.
5. **Second-order weights** (pose, self_collisions, dof_pos_limits, head
   terms, foot terms) show effects of 0.01-0.05 m/s beyond the center-run
   noise but not Lenth's. Treat them as leads. Aliases matter here:
   `upright` is aliased with the linear x angular interaction, `air_time`
   with linear x action_rate, `action_rate_l2` with linear x air_time.
6. **Lenth's threshold is inflated** because the effects are not sparse (two
   huge ones and many medium ones). The center-run threshold is the better
   yardstick here, though it rests on only 4 runs (3 degrees of freedom).

**Weights versus reward form.** Every screening run that walked accurately
on the held-out site (speed error under 0.06 m/s) turned badly (yaw over
0.18 rad/s). C6v2, which changed how tracking is scored (error relative to
the command, smoothed velocity) and what training asks for, reached 0.033
m/s with 0.110 rad/s. Re-weighting moves along the seesaw; changing the
form of the reward moved the seesaw. (C6v2 also trained 1500 iterations,
not 250.)

### What it means for commissioning

The vendor's reward does not pay for accurate walking: a robot standing
still earns about as much speed-tracking reward as one walking. That one
property explains C4 standing still, the slow-move stall, C3 losing
accuracy with more training, and why C6's relative tracking helped so much.
Checking what the reward pays for, cheaply, from the training logs, is a
commissioning step that would have flagged all of it before any site test.

### Next, if wanted

A two-factor follow-up on the seesaw (linear and angular tracking weight,
3 levels each, 2-3 seeds) on top of C6v2, to choose the speed/turn operating
point, or C6v2 with a relative yaw term weighted above the linear one.
