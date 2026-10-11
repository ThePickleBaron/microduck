# Microduck Pre-Training: Deployment-Ready on Delivery

Semester project for **IDAI 600: AI Foundations and Applied Strategy** (VCU, Fall 2026).

New industrial automation (cobots, vision systems, automated guided vehicles) arrives *hardware-ready but not task-ready*: weeks of on-site tuning happen after the money is spent. This project tests a simulation-first alternative at hobby scale. It pre-trains a [Pollen Robotics Microduck](https://pollen-robotics.com/microduck/) walking skill in simulation **before the robot arrives**, then measures how deployment-ready that skill is on a "site" it never trained on.

The question: **how much of commissioning can move before delivery, and what does it take to trust it?**

## The experiment

Four conditions, each evaluated on the same held-out site:

| Key | Condition | What changes | Industrial analog |
| --- | --- | --- | --- |
| c1 | Vendor default | The official shipped walk policy, untouched | Used as delivered |
| c2 | Idealized simulation | Flat floor, all randomization off, ideal actuators | Offline programming against a perfect model |
| c3 | Standard randomization | Pollen's full sim-to-real recipe | Vendor-grade generic robustness |
| c4 | Site-specific pre-training | c3 + rough terrain, gear backlash, 0-100 g payload, wider friction | Simulation tuned to the destination site |

The **readiness gap** is how much worse a policy does on the held-out site (`sites/held_out.toml`: slippery floor, heavy payload, rough floor, worn gears, low battery, slow link, pushes, and all of them combined) than on a nominal site inside its training range. Small gap = ready on delivery. Details: [docs/experiment_protocol.md](docs/experiment_protocol.md).

## Machines

| Machine | Role | GPU |
| --- | --- | --- |
| Desktop | Training | RTX 3090 (CUDA inside WSL2) |
| Laptop | Evaluation, viewing, writing | None needed |
| Hugging Face Jobs | Optional cloud training | Rented L4/A10G/A100 |

Everything runs on Linux. On Windows that means WSL2 Ubuntu 24.04: [docs/windows_wsl_setup.md](docs/windows_wsl_setup.md).

## Setup

```bash
# Windows only, once, in an Administrator PowerShell:
#   .\setup\windows\install_wsl.ps1
# Then inside Ubuntu (clone into ~/, not /mnt/c):
git clone https://github.com/ThePickleBaron/microduck.git ~/microduck-pretraining
cd ~/microduck-pretraining
bash setup/setup.sh          # system libs, uv, Python env, vendor policies, md-check
```

`uv run md-check` should end with "All required checks passed." On the desktop it also lists the 3090.

## Workflow

```bash
# 1. Train (desktop). Same budget for every condition: 4096 envs x 3000 iterations.
uv run md-train c2
uv run md-train c3 --seeds 1 2 3
uv run md-train c4 --seeds 1 2 3
#    Watch progress:  uv run tensorboard --logdir logs/rsl_rl

# 2. Export finished runs to ONNX (policies/c3_seed1.onnx, ...)
uv run python scripts/export_latest.py

# 3. Evaluate every policy on both sites and build the report (any machine)
uv run python scripts/evaluate_all.py           # add --quick for a 2-minute look
#    -> results/report.md, results/report.csv, results/report_gap.png
uv run python scripts/seed_spread.py --md results/seed_spread.md   # per-seed spread, paired comparisons

# 4. Look at a policy walking (needs a display; WSLg provides one on Windows 11)
cd third_party/microduck_rl
uv run scripts/infer_policy.py --walking ../../policies/c4_seed1.onnx --new-cmd-obs
```

Single evaluations: `uv run md-eval --policy policies/c4_seed1.onnx --label c4_seed1 --site sites/held_out.toml`.

After the robot arrives: [docs/hardware_test_protocol.md](docs/hardware_test_protocol.md).

## Commands

| Command | Does |
| --- | --- |
| `md-check` | Verify the install (GPU, display, tasks, policies, logins) |
| `md-conditions` | Print the four conditions |
| `md-train <c2\|c3\|c4>` | Train a condition with the fixed budget; extra flags pass to `train` |
| `md-eval` | Headless CPU evaluation of one ONNX policy on one site |
| `md-report` | Combine results, compute readiness gaps, draw the chart |
| `train`, `play`, `list-envs`, `publish` | The vendored mjlab / Microduck tools |

## Follow-up conditions: C5 and C3x

The first C4 runs learned to shuffle instead of step, and their terrain
curriculum could only move robots down to the easiest level. Two follow-ups
(C1-C4 unchanged):

| Key | Condition | What it is |
| --- | --- | --- |
| c5 | Site fine-tune | C3 policy warm-started, then 1500 iterations on C4's site, with terrain promotion based on progress |
| c3x | Control | Same warm start and 1500 iterations, in C3's own world |

```bash
uv run python scripts/train_followup.py c5 --seeds 1 2 3
uv run python scripts/train_followup.py c3x --seeds 1 2 3
uv run python scripts/train_followup.py --export
```

Details and what to watch: [docs/c5_site_finetune.md](docs/c5_site_finetune.md).

**C6 (careful walk, added 2026-10-10):** C3x plus rewards that make slow
commands count, after the stress tests showed every standard policy stands
still at 0.08-0.15 m/s. `uv run python scripts/train_followup.py c6 --seeds 1 2 3`.
Details: [docs/c6_careful.md](docs/c6_careful.md).

## Side project: steady-cam walking

Separate from the experiment (C1-C4 are untouched): a camera-stabilizing
fine-tune of the walker for a desktop-cameraman duck. It warm-starts from the
C3x run of the same seed (the best walker) and adds camera shake, bob and
horizon-tilt costs at the head camera:

```bash
uv run python scripts/train_steadycam.py            # ~45 min on the 3090
uv run python scripts/train_steadycam.py --export   # -> policies/steadycam_seed1.onnx
uv run python scripts/eval_steadycam.py --policy policies/c3x_seed1.onnx --label c3x          # baseline
uv run python scripts/eval_steadycam.py --policy policies/steadycam_v2_seed1.onnx --label steadycam_v2
```

Details, baseline numbers and what to watch: [docs/steadycam.md](docs/steadycam.md).

## Timelapse videos

Side-by-side videos for non-technical viewers. They show learning over time,
every final policy facing the held-out challenges, and seed-to-seed spread.
They are rebuilt after the fact from the snapshots every run saves every 250
iterations, so **keep `logs/`**.

```bash
uv run python scripts/make_timelapse.py --list          # runs that can be filmed
uv run python scripts/make_timelapse.py all --quick     # ~5 min test -> videos/quick/
uv run python scripts/make_timelapse.py all             # learning, exam, seeds -> videos/
```

Details: [docs/timelapse.md](docs/timelapse.md).

## Stress tests (no training needed)

Readiness vs. training budget (every saved snapshot), each policy's breaking
point under six stresses, and a slow-and-careful site:

```bash
uv run python scripts/stress_tests.py all --quick   # check it works
uv run python scripts/stress_tests.py all           # -> results/stress/*.md, *.png
```

Details: [docs/stress_tests.md](docs/stress_tests.md).

## Deployment check

Can each exported policy be installed on the robot as it is? Shape, packaging
and metadata against the shipped walk policy, numerics, inference cost, and a
stand/walk check in the simulator:

```bash
uv run python scripts/check_policies.py --manifest   # -> results/policy_check.md, policies/manifest.json
```

Details: [docs/deployment_check.md](docs/deployment_check.md).

## Repository layout

```
src/microduck_pretrain/   conditions, task registration, train/eval/report tools
sites/                    nominal.toml, held_out.toml (experiment) and cinema.toml (steady-cam shots)
scripts/                  vendor policy download, export, evaluate-all, HF Jobs shims
setup/                    setup.sh (Linux/WSL) and windows/install_wsl.ps1
docs/                     protocol, Windows setup, cloud training, hardware test, steady-cam
tests/                    pytest suite (uv run pytest -q; -m "not slow" for the fast half)
third_party/microduck_rl/ vendored Pollen Robotics training stack (see VENDORED.md)
policies/                 exported ONNX policies (vendor/ is downloaded, not committed)
results/                  evaluation JSON, report.md, report.csv, report_gap.png
```

## Credits and licenses

The training stack in `third_party/microduck_rl/` is Pollen Robotics' [microduck_rl](https://github.com/pollen-robotics/microduck_rl) (Apache-2.0; 3D model files CC BY-SA-NC), built on [mjlab](https://github.com/mujocolab/mjlab) and Rhoban's [BAM](https://github.com/Rhoban/bam) actuator models. This project's own code is Apache-2.0.

AI use: this repository was scaffolded with Claude (Anthropic) and reviewed by the author, per the IDAI 600 AI-use policy.
