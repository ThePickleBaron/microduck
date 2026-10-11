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
