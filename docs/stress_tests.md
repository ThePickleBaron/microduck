# Stress tests: budget curve, operating envelope, careful site

Added 2026-10-10. Three follow-up experiments on the policies and checkpoints
that already exist: no new training, CPU only. One script runs them all:

```bash
cd ~/microduck-pretraining && git pull
uv run python scripts/stress_tests.py all --quick    # ~5 min: checks everything works
uv run python scripts/stress_tests.py all            # the real thing
```

Results land in `results/stress/` as `<test>.md` (tables), `<test>.csv` and
`<test>.png` (figure). Every simulated run is cached in
`results/stress/cache/`, so re-running only does new work and an interrupted
run resumes. `--workers N` sets parallel processes (default: half the CPU
cores, up to 12). `--conditions c3 c5 c3x` limits which conditions are run.

## 1. Budget curve: readiness vs. training iterations

Every saved snapshot (every 250 iterations) of the C3, C5 and C3x runs, all
seeds, evaluated on the nominal site and the 8 held-out scenarios (1 run of
120 s each). C3 covers iterations 0-3000; C5 and C3x continue from it to 4500,
so one figure shows learning, then fine-tuning on the site vs. simply
training on in C3's world.

Questions it answers: when does readiness stop improving during C3's own
training? Does C5 lose tracking early in fine-tuning or gradually? Had C3x's
gain levelled off by 1500 extra iterations? Industrial analog: how many hours
of on-site commissioning are worth paying for.

About 730 runs.

## 2. Operating envelope: each policy's breaking point

Every exported policy (and C1) on the nominal floor and commands, with one
stress ramped up step by step (60 s x 2 runs per level):

| Stress | Levels |
| --- | --- |
| Foot friction | 1.0 down to 0.15 |
| Payload | 0 to 400 g |
| Command delay | 0 to 120 ms |
| Shoves | 0 to 1.2 m/s, every 2-4 s |
| Battery | 7.4 down to 5.4 V (heavy sag) |
| Floor bumps | 0 to 25 mm |

A level fails if the policy falls 10 or more times per 10 minutes or stalls
for 50% or more of the commanded time. The rating is the last level before the
first failure: a spec sheet for each policy ("rated to 200 g, 60 ms"). It
shows whether a weakness (C2 and delay, for example) is a cliff or a slope.

About 1,440 runs.

## 3. Careful site: slow moves from a standstill

`sites/careful.toml`: 0.08-0.15 m/s walks and 0.3 rad/s turns, each from a
standstill, on the nominal floor and on the combined held-out site. The other
sites command 0.25-0.30 m/s; steady-cam showed some C3x seeds stand still at
slower speeds. Headline metric: stall. Sanity check: C1 (the vendor policy)
stalls 100% here, as expected, since it does not start from rest below about
0.2 m/s.

About 64 runs.

## Time

The quick mode takes about 3 minutes on 2 cloud CPU cores. A full run is
roughly 2,200 simulated runs of 60-120 s: about 2 hours on one core, so about
10-20 minutes with 8-12 workers. It does not touch the GPU, so it can run
during training.

## Note on C1 in the timelapse videos

Until 2026-10-10 the timelapse script used `alpha_walking.onnx` as C1, while
the report and these tests use `velstand.onnx` (Pollen's default walk policy,
as in `docs/experiment_protocol.md`). Fixed in `runs.VENDOR_ONNX`; re-run
`make_timelapse.py exam` to refresh the C1 panels.

## Results, first full run (2026-10-10)

### Budget curve

Mean of 3 seeds, speed error in m/s:

| Iteration | C3 nominal | C3 held-out (mean of 8) |
| --- | --- | --- |
| 0 | 0.142 | 0.160 (476 falls/10 min) |
| 250 | 0.097 | 0.101 |
| 750 | **0.063** | **0.074** |
| 1500 | 0.074 | 0.084 |
| 2999 | 0.099 | 0.102 |

- C3 tracks best around iteration 750-1000 and then gets steadily worse
  (+55% nominal error by 3000), while falls stay near zero throughout. The
  later iterations widen randomization and head commands; whatever they buy
  is not measured by these sites.
- C3x's gain over C3 comes in the first 250 extra iterations (held-out 0.102
  -> 0.095) and then plateaus (0.096 at 4499).
- C5 gets worse in its first 250 iterations (0.102 -> 0.112) and never
  recovers (0.112 at 4499; combined scenario 0.102 -> 0.118).

### Operating envelope

60 s x 2 runs per level; rating = last level before failing:

| Policy | Friction | Payload | Delay | Shove | Battery | Bumps |
| --- | --- | --- | --- | --- | --- | --- |
| C1 (vendor) | 0.15+ | 400 g+ | 60 ms | 0.4 m/s | 5.4 V+ | 25 mm+ |
| C2 (3 seeds) | 0.4 | 150-200 g | 20 ms | fails at 0.2 | 6.6 V (2 of 3) | 4-8 mm |
| C3 | 0.15-0.2 | 400 g+ | 40-60 ms | 0.4 | 5.4 V+ | 20-25 mm+ |
| C4 | fails at start (stands still) | | | | | |
| C5 | 0.15+ (all 3) | 400 g+ | 60 ms (all 3) | 0.4 | 5.4 V+ | 25 mm+ (all 3) |
| C3x | 0.15-0.2 | 400 g+ | 40-60 ms | 0.4 | 5.4 V+ | 25 mm+ |

- Every randomized policy hits the same two walls: command delay of 60-80 ms
  and shoves of 0.4-0.6 m/s. These look like limits of the robot and its
  50 Hz control, not of the training recipe.
- C2 is the most accurate on calm floors but has the narrowest envelope on
  every axis.
- C5 has the widest and most consistent envelope: all three seeds reach the
  end of the friction and bump ranges, and it falls least at 0.6-0.8 m/s
  shoves. Site training bought robustness at the edges while costing tracking
  accuracy, which the original held-out metric did not show.

### Careful site

| Condition | Stalled on slow moves (home / site) |
| --- | --- |
| C1, C3, C4, C5, C3x (all seeds) | 97-100% / 97-100% |
| C2 | 81% / 61% |

| Steady-cam v2 (positive control) | 32% (4-63) / 28% (22-38) |

Every policy trained with Pollen's randomization, and the vendor policy,
stands still when asked for 0.08-0.15 m/s from a standstill. C2 moves part of
the time. The held-out site (0.25-0.30 m/s) could not see this.

Positive control: steady-cam v2, warm-started from C3x and trained with a
stall penalty, stalls 4-63% (mean about 30%) where its C3x parents stall
97-100%, and its speed error halves (0.066 -> 0.034 m/s). So the test does
detect slow walking, and a stall penalty largely closes the gap. Seed 3 is
the weakest (63% at home), matching its parent being the most stall-prone at
cinematic speeds. Its yaw error is higher than the standing policies'
(0.07-0.10 vs 0.05 rad/s): it turns, but less precisely.
