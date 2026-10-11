"""Reward-weight screening experiment (added 2026-10-11).

Which of the vendor reward's 15 active weights move deployment readiness?
A two-level Plackett-Burman screen (16 runs, each weight at 0.5x or 2x its
vendor value, 15 columns of a 16x16 Hadamard matrix) plus 4 center runs at
the vendor weights (training seeds 1-4) to measure run-to-run noise.

Every run warm-starts from the finished C3x seed 1 policy and trains a short
burst (SCREEN_ITERATIONS) in C3x's world with curricula at their final stage,
so a run measures how the policy shifts when the weights shift. That is a
local sensitivity around the vendor recipe, not what training from scratch
with those weights would produce.

One task id serves every run: the weight multipliers come from the
MICRODUCK_REWARD_SCALES environment variable (JSON {term: multiplier}),
read when the task registers inside each training process. Without it the
task is exactly C3x. Driver: scripts/reward_screen.py. Write-up:
docs/reward_screen.md.

Importing this module is light (numpy only): followups.register() calls
register() once C3x exists, so the evaluation workers of the driver never
pull in mjlab.
"""

from __future__ import annotations

import copy
import json
import os

import numpy as np

SCREEN_TASK = "Pretrain-RewardScreen-Flat-MicroDuck"
SCREEN_EXPERIMENT = "reward_screen"
SCALES_ENV = "MICRODUCK_REWARD_SCALES"
SCREEN_ITERATIONS = 250
LOW, HIGH = 0.5, 2.0
N_CENTER = 4
PARENT_KEY, PARENT_SEED = "c3x", 1

# The vendor reward's active terms (body_pose_tracking has weight 0 and is
# left out), in a fixed order: column j of the design is FACTORS[j].
FACTORS = [
    "track_linear_velocity",
    "track_angular_velocity",
    "upright",
    "pose",
    "body_ang_vel",
    "angular_momentum",
    "dof_pos_limits",
    "action_rate_l2",
    "air_time",
    "foot_clearance",
    "foot_swing_height",
    "foot_slip",
    "self_collisions",
    "head_pose_tracking",
    "head_pose_bias",
]


def hadamard(n: int) -> np.ndarray:
    """Sylvester Hadamard matrix, n a power of two."""
    h = np.array([[1]])
    while h.shape[0] < n:
        h = np.block([[h, h], [h, -h]])
    if h.shape[0] != n:
        raise ValueError("n must be a power of two")
    return h


def design_matrix() -> np.ndarray:
    """16 runs x 15 factors of +-1: the Hadamard matrix without its all-ones
    column. Columns are balanced (8 high, 8 low) and mutually orthogonal, so
    each main effect is estimated from all 16 runs (resolution III: main
    effects are aliased with two-factor interactions)."""
    return hadamard(16)[:, 1:]


def runs() -> list[dict]:
    """The run table, in execution order: center runs spread through the
    sequence so drift (thermal, software) would show up in them."""
    x = design_matrix()
    pb = [{"run": f"pb{i + 1:02d}", "kind": "screen", "seed": PARENT_SEED,
           "levels": {f: int(x[i, j]) for j, f in enumerate(FACTORS)}} for i in range(len(x))]
    ctr = [{"run": f"ctr{k + 1}", "kind": "center", "seed": k + 1,
            "levels": {f: 0 for f in FACTORS}} for k in range(N_CENTER)]
    order = [ctr[0], *pb[0:5], ctr[1], *pb[5:10], ctr[2], *pb[10:16], ctr[3]]
    for r in order:
        r["scales"] = {f: (HIGH if v > 0 else LOW if v < 0 else 1.0) for f, v in r["levels"].items()}
    return order


def scales_from_env() -> dict[str, float]:
    raw = os.environ.get(SCALES_ENV, "").strip()
    if not raw:
        return {}
    scales = {k: float(v) for k, v in json.loads(raw).items()}
    unknown = set(scales) - set(FACTORS)
    if unknown:
        raise ValueError(f"{SCALES_ENV}: unknown reward terms {sorted(unknown)}")
    return scales


def apply_scales(cfg, scales: dict[str, float]):
    for name, mult in scales.items():
        if name not in cfg.rewards:
            raise KeyError(f"reward term {name!r} not in this task")
        cfg.rewards[name].weight = cfg.rewards[name].weight * mult
    return cfg


def make_screen_env_cfg(play: bool = False):
    from microduck_pretrain.followups import make_c3x_env_cfg

    return apply_scales(make_c3x_env_cfg(play=play), scales_from_env())


def _rl_cfg():
    from mjlab_microduck.tasks import microduck_velocity_env_cfg as _vel

    cfg = copy.deepcopy(_vel.MicroduckRlCfg)
    cfg.experiment_name = SCREEN_EXPERIMENT
    cfg.run_name = SCREEN_EXPERIMENT
    cfg.wandb_project = "microduck_pretrain"
    cfg.max_iterations = SCREEN_ITERATIONS
    return cfg


def register() -> None:
    from mjlab.tasks.registry import list_tasks, register_mjlab_task

    import mjlab_microduck.tasks as _md_tasks

    if SCREEN_TASK in set(list_tasks()):
        return
    scales = scales_from_env()
    if scales:
        print(f"[reward_screen] weight multipliers: {json.dumps(scales, sort_keys=True)}", flush=True)
    register_mjlab_task(
        task_id=SCREEN_TASK,
        env_cfg=make_screen_env_cfg(),
        play_env_cfg=make_screen_env_cfg(play=True),
        rl_cfg=_rl_cfg(),
        runner_cls=_md_tasks.MicroduckOnPolicyRunner,
    )

