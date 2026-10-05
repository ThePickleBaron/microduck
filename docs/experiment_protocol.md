# Experiment protocol

## Question

How deployment-ready can a robot skill be on delivery day if it is trained only in simulation, and what would an organization need to see before trusting it?

1. **Transfer.** How much performance survives the move from the training simulator to a different, held-out environment? Simulation-to-simulation now; simulation-to-real after delivery.
2. **Randomization.** Does training on broader variation improve readiness, and at what cost in compute and peak performance?
3. **Commissioning shift.** How much tuning can move before delivery, measured on the real robot as time from unboxing to a custom skill running?

## Conditions

Defined once in `src/microduck_pretrain/conditions.py` and built in `src/microduck_pretrain/tasks.py` from the vendored velocity task, so the reward recipe, the 61-D observation layout and the ONNX export match the shipped policies exactly. Only the randomization and the environment differ.

| Key | Task id | Training environment |
| --- | --- | --- |
| c1 | (none) | Pollen's shipped `velstand.onnx`, the robot's default walk policy |
| c2 | `Pretrain-C2-Idealized-Flat-MicroDuck` | Flat; every `ENABLE_*` toggle off; base pushes and CoM offset removed; BAM at a fixed 7.4 V with no sag or delay; foot friction 1.0; no observation noise |
| c3 | `Pretrain-C3-Standard-Flat-MicroDuck` | Identical to `Mjlab-Velocity-Flat-MicroDuck` (Pollen's full recipe) |
| c4 | `Pretrain-C4-Site-Rough-Backlash-MicroDuck` | c3 on the rough-terrain generator, walk-model backlash robot (+/-1 deg per servo), 0-100 g trunk payload, foot friction 0.5-1.5 |

## Fair comparison

- Same budget for every trained condition: 4096 parallel environments x 3000 PPO iterations (`md-train` defaults). Change both with `--num-envs` / `--iterations`, but change them for every condition.
- Three seeds per trained condition (`--seeds 1 2 3`). Report the mean and the spread.
- Evaluate the **last** checkpoint of every run (`scripts/export_latest.py`); no cherry-picking.
- Record training wall-clock and GPU from the TensorBoard/W&B run for the compute-cost metric.

## Evaluation

`md-eval` runs an ONNX policy in **CPU MuJoCo** with the same BAM servo model and solver settings used in training. Training runs on GPU MuJoCo Warp, so this is a different simulator, a miniature sim-to-real gap.

Each scenario repeats a command schedule (forward, turn, diagonal, stop, restart, backward, turn back) for 120 s, three seeds. A fall (trunk below 6 cm or tilted past 60 deg) is counted and the robot reset.

| Metric | Meaning |
| --- | --- |
| `falls_per_10min` | Robustness |
| `lin_vel_err_mps` | Planar speed error, 1 s moving average, while upright |
| `yaw_rate_err_radps` | Turning error, same averaging |
| `stall_fraction` | Share of commanded-motion time spent standing still (achieved < 20% of the command) |
| `upright_fraction` | Share of time not recovering from a fall |

**Sites.** `sites/nominal.toml` stays inside every training range. `sites/held_out.toml` pushes one factor at a time just beyond the widest range any condition trained on (c4's), then combines them in `site_combined`. Edit or add scenarios freely; `tests/test_sites.py` guards the "held out" property.

**Readiness gap** (`md-report`): tracking gap % = (held-out speed error - nominal speed error) / nominal; plus extra falls per 10 min and extra stall percentage points.

## Early observation (simulation-to-simulation)

Running the shipped policies through `md-eval` while building this repo:

- Started from a standstill, the shipped walk policies **stay standing for commands up to about 0.2 m/s** in the CPU simulator, then walk at 0.3 m/s. Once moving, they keep walking at lower commands. The schedule therefore starts and restarts at 0.3 m/s, and `stall_fraction` tracks the effect.
- Achieved speed is roughly half the commanded speed in the CPU simulator.

Neither has been checked on hardware. Whether these are artifacts of the CPU simulator or real behaviors is exactly the kind of sim-to-real question the hardware test answers, and a useful Topic 5/6 discussion point.

## Generative AI's role

Gemini (or another assistant) drafts edge-case scenarios for the site files, acceptance criteria, and summaries of `results/report.csv`. Keep its suggestions and your own side by side in the AI Strategy Development File; that comparison is a Topic 5 data point.
