"""Follow-up conditions added after the first C1-C4 results (2026-10-09).

Why: the original C4 (site-specific training from scratch) learned to shuffle
instead of step, and its terrain curriculum never left level 0 because mjlab's
rule promotes a robot only after it ends an episode half a tile (4 m) from its
start, which a 0.4 m/s duck with changing commands almost never does. See
docs/c5_site_finetune.md.

C5  - site fine-tune: warm-start from a finished C3 policy (seed N from C3
      seed N) and fine-tune on C4's site conditions (rough terrain, backlash,
      payload, wide friction) with a terrain curriculum that promotes on
      progress along the commanded direction.
C3x - control: the same warm start and the same number of extra iterations,
      but in C3's own world. Separates "site-specific" from "trained longer".

Neither changes C1-C4: both are new tasks with their own log folders. Train
them with scripts/train_followup.py.
"""

from __future__ import annotations

import copy

import torch
from mjlab.managers import CurriculumTermCfg
from mjlab.managers.metrics_manager import MetricsTermCfg

import mjlab_microduck.tasks as _md_tasks
from mjlab_microduck.tasks import microduck_velocity_env_cfg as _vel
from microduck_pretrain import tasks as _tasks
from microduck_pretrain.curricula import _collapse_curricula_to_final

C5_TASK = "Pretrain-C5-SiteFinetune-Rough-Backlash-MicroDuck"
C5_EXPERIMENT = "c5_site_finetune"
C3X_TASK = "Pretrain-C3X-Standard-Extended-Flat-MicroDuck"
C3X_EXPERIMENT = "c3x_extended"
FINETUNE_ITERATIONS = 1500

# Terrain promotion on progress (fractions of the distance the commands asked
# for, measured along the commanded direction).
PROMOTE_FRAC = 0.7
DEMOTE_FRAC = 0.3
MIN_EXPECTED_M = 1.0   # below this the episode asked for too little motion to judge


# --------------------------------------------------------------------------
# Progress tracking: expected world displacement from the commands
# --------------------------------------------------------------------------
def _expected_buffer(env) -> torch.Tensor:
    if not hasattr(env, "_md_expected_disp"):
        env._md_expected_disp = torch.zeros(env.num_envs, 2, device=env.device)
    return env._md_expected_disp


def _yaw(quat_wxyz: torch.Tensor) -> torch.Tensor:
    w, x, y, z = quat_wxyz.unbind(-1)
    return torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def commanded_progress(env, command_name: str = "twist") -> torch.Tensor:
    """Metrics term, runs every policy step.

    Integrates the displacement the twist commands asked for, rotating the
    body-frame command by the robot's actual heading (so a robot that turns
    late is judged on translation only). Returns progress along that
    direction as a fraction of the requested distance (0 when too little was
    requested yet), which also makes it a useful TensorBoard curve:
    Episode_Metrics/commanded_progress.
    """
    asset = env.scene["robot"]
    buf = _expected_buffer(env)
    cmd = env.command_manager.get_command(command_name)[:, :2]
    yaw = _yaw(asset.data.root_link_quat_w)
    c, s = torch.cos(yaw), torch.sin(yaw)
    world = torch.stack((c * cmd[:, 0] - s * cmd[:, 1], s * cmd[:, 0] + c * cmd[:, 1]), dim=-1)
    buf += world * float(env.step_dt)

    actual = asset.data.root_link_pos_w[:, :2] - env.scene.env_origins[:, :2]
    exp_len = buf.norm(dim=-1)
    along = (actual * buf).sum(dim=-1) / exp_len.clamp(min=1e-6)
    ratio = torch.where(exp_len >= MIN_EXPECTED_M, along / exp_len.clamp(min=1e-6), torch.zeros_like(exp_len))
    return torch.nan_to_num(ratio.clamp(-1.0, 2.0), nan=0.0)


def terrain_levels_progress(
    env,
    env_ids: torch.Tensor,
    command_name: str = "twist",
    promote_frac: float = PROMOTE_FRAC,
    demote_frac: float = DEMOTE_FRAC,
    min_expected_m: float = MIN_EXPECTED_M,
) -> dict[str, torch.Tensor]:
    """Curriculum term (runs at reset, before the scene resets).

    Promote a robot that covered at least `promote_frac` of the commanded
    distance along the commanded direction; demote one that covered less than
    `demote_frac`. Episodes that asked for under `min_expected_m` (standing
    robots, dithering commands) neither promote nor demote. Replaces mjlab's
    fixed half-tile (4 m) rule, which this robot almost never satisfies.
    """
    del command_name
    terrain = env.scene.terrain
    buf = _expected_buffer(env)
    if terrain is not None and terrain.cfg.terrain_generator is not None:
        asset = env.scene["robot"]
        actual = asset.data.root_link_pos_w[env_ids, :2] - env.scene.env_origins[env_ids, :2]
        exp = buf[env_ids]
        exp_len = exp.norm(dim=-1)
        along = (actual * exp).sum(dim=-1) / exp_len.clamp(min=1e-6)
        judged = exp_len >= min_expected_m
        move_up = judged & (along >= promote_frac * exp_len)
        move_down = judged & (along < demote_frac * exp_len) & ~move_up
        terrain.update_env_origins(env_ids, move_up, move_down)
    buf[env_ids] = 0.0
    if terrain is None or not hasattr(terrain, "terrain_levels"):
        z = torch.zeros((), device=env.device)
        return {"mean": z, "max": z}
    levels = terrain.terrain_levels.float()
    return {"mean": levels.mean(), "max": levels.max()}


# --------------------------------------------------------------------------
# Env configs
# --------------------------------------------------------------------------
def make_c5_env_cfg(play: bool = False):
    cfg = _tasks.make_site_env_cfg(play=play)
    # The warm-started C3 policy was trained under the final stage of every
    # curriculum; keep it there instead of restarting them at stage 0.
    _collapse_curricula_to_final(cfg)
    if "terrain_levels" in cfg.curriculum:
        cfg.curriculum["terrain_levels"] = CurriculumTermCfg(
            func=terrain_levels_progress, params={"command_name": "twist"}
        )
    cfg.metrics["commanded_progress"] = MetricsTermCfg(
        func=commanded_progress, params={"command_name": "twist"}
    )
    return cfg


def make_c3x_env_cfg(play: bool = False):
    cfg = _tasks.make_standard_env_cfg(play=play)
    _collapse_curricula_to_final(cfg)
    cfg.metrics["commanded_progress"] = MetricsTermCfg(
        func=commanded_progress, params={"command_name": "twist"}
    )
    return cfg


def _rl_cfg(experiment: str):
    cfg = copy.deepcopy(_vel.MicroduckRlCfg)
    cfg.experiment_name = experiment
    cfg.run_name = experiment
    cfg.wandb_project = "microduck_pretrain"
    cfg.max_iterations = FINETUNE_ITERATIONS
    return cfg


FOLLOWUPS = {
    # key: (task id, experiment folder, cfg factory)
    "c5": (C5_TASK, C5_EXPERIMENT, make_c5_env_cfg),
    "c3x": (C3X_TASK, C3X_EXPERIMENT, make_c3x_env_cfg),
}


def register() -> None:
    from mjlab.tasks.registry import list_tasks, register_mjlab_task

    have = set(list_tasks())
    for task_id, experiment, make in FOLLOWUPS.values():
        if task_id in have:
            continue
        register_mjlab_task(
            task_id=task_id,
            env_cfg=make(),
            play_env_cfg=make(play=True),
            rl_cfg=_rl_cfg(experiment),
            runner_cls=_md_tasks.MicroduckOnPolicyRunner,
        )


# Self-register when loaded (same reason as steadycam.py: `import mjlab` loads
# the task plugins, which import this module, possibly before it finishes).
register()
