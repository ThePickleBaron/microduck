# Reward surface: walking-speed vs turning emphasis on C6v2

Added 2026-10-11, after the reward screen (`docs/reward_screen.md`).

## Why

The screen found that the two tracking weights form a seesaw: more weight on
walking speed gives accurate walking and poor turning, and the reverse. C6v2
(the best walker so far) sits at one point on its own seesaw: held-out speed
error 0.033 m/s, yaw error 0.110 rad/s, against C3x's 0.096 / 0.085. This
study maps C6v2's seesaw so the speed/turn balance is chosen, not inherited.

## Design

| | |
| --- | --- |
| Factor A, "linear" | multiplier on `track_linear_velocity` (vendor, 2.0) and `track_lin_relative` (C6, 1.0) |
| Factor B, "turning" | multiplier on `track_angular_velocity` (vendor, 2.0) and `track_yaw_relative` (C6v2, 1.0) |
| Levels | 0.5x, 1x, 2x (coded -1, 0, +1 on a log2 scale) |
| Design | 3x3 full factorial, 2 replicates = 18 runs, in a fixed random order |
| Replicates | seed N warm-starts from C6v2 seed N and trains with seed N (N = 1, 2), so replicate spread includes parent-to-parent variation |
| Training per run | 250 iterations in C6v2's world (its commands, rewards, curricula at final stage), about 9 min |
| Scoring | as the screen: nominal, 8 held-out scenarios, careful site; 1 run x 120 s each |
| Model | full quadratic per response, y = b0 + b1 a + b2 b + b12 ab + b11 a^2 + b22 b^2, least squares (12 residual df) |
| Checks | standard errors and R^2; lack of fit against the replicate error (3 and 9 df) |

Each factor scales a pair because C6v2 tracks speed and turning twice (the
vendor's absolute terms and C6's relative ones); scaling one of each pair
would leave the other in charge. The 1x/1x cell is C6v2 plus 250 more
iterations, the in-study baseline.

**Operating point.** Minimize J = max(held-out speed error / 0.033,
held-out yaw error / 0.085) on the fitted surfaces, inside the tested
region, with predicted careful-site stall at or under 5% and held-out falls
at or under 1 per 10 min. J = 1.0 would be C6v2's walking accuracy and C3x's
turning accuracy at once. Both targets are fixed in advance from earlier
results, not chosen after seeing this data.

**Confirmation.** The predicted point is a hypothesis. If it beats C6v2, the
next step is a full run at that point from the C3 parents (1500 iterations,
3 seeds, the same budget as C6/C6v2), evaluated like every other condition.

## Limits

Short fine-tunes from C6v2, not training from scratch. Two seeds per cell.
Three levels per factor, so the model cannot see anything finer than a
quadratic; the lack-of-fit test says when that matters. The model is not
trusted outside the 0.5x-2x square.

## Run it

```bash
cd ~/microduck-pretraining && git pull && bash scripts/reward_surface_batch.sh
```

Or step by step: `uv run python scripts/reward_surface.py design | train | evaluate | analyze`.
Outputs in `results/reward_surface/`: `surface.md` (cell means, fitted
coefficients, operating point), `surface.png` (contours of each response,
the objective with the stall/falls limit, and the speed-vs-yaw trade-off),
`runs.csv`. TensorBoard: `--logdir logs/rsl_rl/reward_surface`.

## Results (2026-10-12)

Held-out speed / yaw error, mean of 2 seeds per cell (C6v2 parents:
0.031 / 0.106 and 0.035 / 0.093):

| turning \ linear | 0.5x | 1x | 2x |
| --- | --- | --- | --- |
| 2x | 0.038 / 0.070 | **0.031 / 0.080** | 0.029 / 0.088 |
| 1x | 0.035 / 0.101 | 0.029 / 0.101 | 0.029 / 0.128 |
| 0.5x | 0.034 / 0.141 | 0.029 / 0.152 | 0.025 / 0.142 |

- **Operating point: linear 1x, turning 2x.** Held-out speed 0.031 m/s, yaw
  0.080 rad/s (seed 1: 0.031 / 0.082; seed 2: 0.032 / 0.079), J = 0.95 against a
  predicted 1.29 for C6v2 as is. That is C6v2's walking accuracy with turning
  better than C3x's 0.085. Careful-site stall 0%, falls 0.6 per 10 min on both
  seeds. The fitted optimum and the best tested cell agree.
- **On C6v2 the seesaw is lopsided.** Each doubling of the turning pair
  lowers held-out yaw error by 0.033 rad/s (significant), while held-out mean
  speed error moves by +0.002 m/s (not significant). Under the vendor reward
  (the screen) the same change cost much more speed. The relative speed term
  seems to hold walking accuracy in place. Each doubling of the linear pair
  lowers speed error by 0.004 m/s (significant) and raises yaw error by 0.008
  (not significant).
- **More training alone does not fix turning:** the 1x/1x cell (C6v2 + 250
  iterations) stays at 0.029 / 0.101.
- **Model checks:** no response shows significant lack of fit (p 0.23-0.97).
  R^2 is 0.83 for held-out yaw, 0.57-0.74 for the speed responses (small
  effects against seed noise), and 0.33-0.38 for stall and falls, which are
  nearly flat (stall 0-0.7%, falls 0-1.9 per 10 min). The falls limit therefore
  rests on a noisy fit; it does not affect the chosen point (observed 0.6).
- **Caveat:** the optimum is on the edge of the tested square, and yaw error
  was still falling at 2x, so the best setting may lie beyond 2x. The model
  is not used outside the square. All runs are 250-iteration fine-tunes.

**Next:** a confirmation run, C6v3 = C6v2 with the turning pair at 2x,
trained from the C3 parents (1500 iterations, 3 seeds) and evaluated like every
other condition; optionally a 3x arm to test whether turning keeps improving.
