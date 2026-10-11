"""Two-weight response-surface study on top of C6v2 (added 2026-10-11).

The reward screen (docs/reward_screen.md) found that walking-speed accuracy
and turning accuracy are a seesaw set by two weights, and that changing the
reward's form (C6v2) beat any weight setting. This study maps that seesaw for
C6v2 itself, to choose the speed/turn operating point on purpose.

Factors (each a multiplier on a PAIR of terms, so "how much the reward cares
about walking speed" vs "how much it cares about turning"):

    A  linear   track_linear_velocity (vendor) + track_lin_relative (C6)
    B  turning  track_angular_velocity (vendor) + track_yaw_relative (C6)

Levels 0.5x, 1x, 2x (log2 coded -1, 0, +1): a 3x3 full factorial, 2 replicates
(seed N warm-starts from C6v2 seed N and trains with seed N), 18 runs in a
fixed random order. Each run trains SURFACE_ITERATIONS in C6v2's world.
Fitted model per response: full quadratic in the coded factors.

Same mechanism as the screen: one task id, multipliers from
MICRODUCK_SURFACE_SCALES (its own variable, so the screen task never sees
these terms), registered from followups.register(). Driver:
scripts/reward_surface.py. Write-up: docs/reward_surface.md.
"""

from __future__ import annotations

import copy
import json
import os

import numpy as np

SURFACE_TASK = "Pretrain-RewardSurface-C6v2-Flat-MicroDuck"
SURFACE_EXPERIMENT = "reward_surface"
SCALES_ENV = "MICRODUCK_SURFACE_SCALES"
SURFACE_ITERATIONS = 250
PARENT_KEY = "c6v2"
SEEDS = (1, 2)
LEVELS = (-1, 0, 1)            # log2 of the multiplier
ORDER_SEED = 2026

FACTORS = {
    "linear": ("track_linear_velocity", "track_lin_relative"),
    "turning": ("track_angular_velocity", "track_yaw_relative"),
}
TERMS = [t for pair in FACTORS.values() for t in pair]


def _code(x: int) -> str:
    return {-1: "m", 0: "0", 1: "p"}[x]


def runs() -> list[dict]:
    """The 18 runs in execution order (a fixed random permutation, so drift
    over the evening cannot line up with either factor)."""
    rows = []
    for seed in SEEDS:
        for a in LEVELS:
            for b in LEVELS:
                scales = {t: 2.0 ** a for t in FACTORS["linear"]} | {t: 2.0 ** b for t in FACTORS["turning"]}
                rows.append({"run": f"L{_code(a)}T{_code(b)}_s{seed}", "seed": seed, "parent_seed": seed,
                             "x": {"linear": a, "turning": b}, "scales": scales})
    order = np.random.default_rng(ORDER_SEED).permutation(len(rows))
    return [rows[i] for i in order]


def scales_from_env() -> dict[str, float]:
    raw = os.environ.get(SCALES_ENV, "").strip()
    if not raw:
        return {}
    scales = {k: float(v) for k, v in json.loads(raw).items()}
    unknown = set(scales) - set(TERMS)
    if unknown:
        raise ValueError(f"{SCALES_ENV}: not a factor of this study: {sorted(unknown)}")
    return scales


def make_surface_env_cfg(play: bool = False):
    from microduck_pretrain.followups import make_c6v2_env_cfg
    from microduck_pretrain.reward_screen import apply_scales

    return apply_scales(make_c6v2_env_cfg(play=play), scales_from_env())


def _rl_cfg():
    from mjlab_microduck.tasks import microduck_velocity_env_cfg as _vel

    cfg = copy.deepcopy(_vel.MicroduckRlCfg)
    cfg.experiment_name = SURFACE_EXPERIMENT
    cfg.run_name = SURFACE_EXPERIMENT
    cfg.wandb_project = "microduck_pretrain"
    cfg.max_iterations = SURFACE_ITERATIONS
    return cfg


def register() -> None:
    from mjlab.tasks.registry import list_tasks, register_mjlab_task

    import mjlab_microduck.tasks as _md_tasks

    if SURFACE_TASK in set(list_tasks()):
        return
    scales = scales_from_env()
    if scales:
        print(f"[reward_surface] weight multipliers: {json.dumps(scales, sort_keys=True)}", flush=True)
    register_mjlab_task(
        task_id=SURFACE_TASK,
        env_cfg=make_surface_env_cfg(),
        play_env_cfg=make_surface_env_cfg(play=True),
        rl_cfg=_rl_cfg(),
        runner_cls=_md_tasks.MicroduckOnPolicyRunner,
    )
