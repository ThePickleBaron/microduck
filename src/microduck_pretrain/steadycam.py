"""Steady-cam walking: a camera-stabilizing fine-tune of the C3 walking policy.

A side project next to the semester experiment (it does not touch C1-C4). The
goal is a "desktop cameraman": the duck walks slow dolly, truck and pan moves
while the head camera stays as steady as a hand-held gimbal.

How it trains
-------------
Warm start from a finished C3 (standard randomization) checkpoint, keep the
proven walking recipe, and add three camera costs measured at the
``head_camera`` site in the world frame:

* ``camera_ang_rate``  - angular velocity of the camera, minus the yaw rate the
  operator commanded (a commanded pan is wanted motion, not shake).
* ``camera_lin_acc``   - linear acceleration of the camera (the step-by-step
  bob and jolt a viewer sees as shake). Finite-differenced between policy
  steps; clipped so a velocity push cannot dominate a batch.
* ``camera_roll``      - tilt of the image horizon.

Following the vendored AGENTS.md lessons:

* Only the *escapable* part is priced. The head is ~38% of the robot's mass and
  must oscillate a little while walking; the neck (4 DOF) can counter-rotate
  and the gait can be made smoother, but not to zero. Weights are sized so the
  camera stack costs a fraction of the walking reward, never enough to make
  "stand still" the best policy.
* Costs are introduced on a curriculum from 0, after the skill exists (it does:
  the run starts from a trained walker), so nothing taxes walking cold.
* The instantaneous head-pose tracking reward is lowered (2.0 -> 0.5) and the
  1 s EMA head-bias penalty is kept at full weight: the head must point where
  it is told *on average*, and is free to deviate for a fraction of a second to
  cancel body motion. That deviation is the stabilization.
* The 61-D observation layout and the command block are untouched, so the
  exported ONNX is hot-swappable with every other Microduck walking policy.

Commands are narrowed to cinematic speeds (slow dollies, gentle pans).
"""

from __future__ import annotations

import copy

import torch
from mjlab.managers import CurriculumTermCfg, RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import quat_apply

import mjlab_microduck.tasks as _md_tasks
from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks import microduck_velocity_env_cfg as _vel
from microduck_pretrain.curricula import _collapse_curricula_to_final  # noqa: F401 (re-exported)

CAMERA_SITE = "head_camera"
# In the head_camera site frame, +Y is the image's left-right axis: on the walk
# model with every joint at zero the site frame is aligned with the world
# (+X forward along the head, +Y left, +Z up). Locked by tests/test_steadycam.py.
CAMERA_LATERAL_AXIS = (0.0, 1.0, 0.0)

TASK_ID = "Steadycam-Walk-Flat-MicroDuck"
EXPERIMENT = "steadycam_walk"
# v2 (2026-10-10): v1 learned to stand still. See make_steadycam_v2_env_cfg.
TASK_ID_V2 = "Steadycam-Walk-v2-Flat-MicroDuck"
EXPERIMENT_V2 = "steadycam_walk_v2"
DEFAULT_ITERATIONS = 1500

# Cinematic command ranges (m/s, m/s, rad/s): slow dolly, truck and pan moves.
CINE_LIN_VEL_X = (-0.20, 0.25)
CINE_LIN_VEL_Y = (-0.12, 0.12)
CINE_ANG_VEL_Z = (-0.6, 0.6)
# Pushes stay on for robustness but softer: a desk is not a hallway.
CINE_PUSH_RANGE = (-0.15, 0.15)

# Final weights of the camera costs (negative: they are costs >= 0).
# Calibrated on the vendor walk policy at 0.3 m/s in the CPU evaluator
# (scripts/eval_steadycam.py): camera shake ~69 deg/s RMS (1.2 rad/s), bob
# ~4.6 m/s^2 RMS, horizon tilt ~4.6 deg RMS. At those levels the costs are
# ~0.22 + ~0.63 + ~0.06 = ~0.9 per second, about 10% of a trained walker's
# positive reward (~8-10 per second): a real gradient that cannot make
# standing still the better deal (tracking + air_time alone pay ~5/s).
CAMERA_WEIGHTS = {
    "camera_ang_rate": -0.15,
    "camera_lin_acc": -0.03,
    "camera_roll": -10.0,
}
# Ramp each cost in over the first 600 iterations (0 -> 1/3 -> 2/3 -> full).
RAMP_ITERS = (0, 150, 350, 600)
HEAD_TRACKING_WEIGHT = 0.5
ANG_RATE_CLIP = 25.0          # (rad/s)^2  - caps a fall/push spike
LIN_ACC_CLIP = 100.0          # (m/s^2)^2


# --------------------------------------------------------------------------
# Reward terms (all return costs >= 0 -> use NEGATIVE weights)
# --------------------------------------------------------------------------
def _camera_site_id(env, asset) -> int:
    if not hasattr(env, "_steadycam_site_idx"):
        ids, _ = asset.find_sites(CAMERA_SITE)
        if len(ids) != 1:
            raise RuntimeError(f"expected one '{CAMERA_SITE}' site, found {len(ids)}")
        env._steadycam_site_idx = int(ids[0])
    return env._steadycam_site_idx


def camera_ang_rate_cost(
    env,
    command_name: str = "twist",
    clip: float = ANG_RATE_CLIP,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """|omega_cam - commanded yaw rate|^2 in the world frame. Cost >= 0."""
    asset = env.scene[asset_cfg.name]
    sid = _camera_site_id(env, asset)
    omega = asset.data.site_ang_vel_w[:, sid].clone()  # (N, 3)
    cmd = env.command_manager.get_command(command_name)  # (N, 3): vx, vy, wz
    omega[:, 2] = omega[:, 2] - cmd[:, 2]
    cost = (omega**2).sum(dim=-1)
    return torch.nan_to_num(cost, nan=0.0).clamp(max=clip)


def camera_lin_acc_cost(
    env,
    clip: float = LIN_ACC_CLIP,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """|d v_cam / dt|^2 between policy steps (world frame). Cost >= 0.

    Stateful: keeps the previous camera velocity per env. A freshly reset env
    has no previous sample, so it costs 0 on its first step.
    """
    asset = env.scene[asset_cfg.name]
    sid = _camera_site_id(env, asset)
    vel = asset.data.site_lin_vel_w[:, sid]  # (N, 3)
    if not hasattr(env, "_steadycam_prev_vel"):
        env._steadycam_prev_vel = vel.clone()
    fresh = env.episode_length_buf <= 1
    env._steadycam_prev_vel[fresh] = vel[fresh]
    acc = (vel - env._steadycam_prev_vel) / float(env.step_dt)
    env._steadycam_prev_vel = vel.clone()
    cost = (acc**2).sum(dim=-1)
    return torch.nan_to_num(cost, nan=0.0).clamp(max=clip)


def camera_roll_cost(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Horizon tilt: squared world-z component of the image's lateral axis.

    0 when the horizon is level, sin^2(roll) otherwise. Pitching the camera up
    or down (aiming) costs nothing. Cost >= 0.
    """
    asset = env.scene[asset_cfg.name]
    sid = _camera_site_id(env, asset)
    quat = asset.data.site_quat_w[:, sid]  # (N, 4) wxyz
    lateral = torch.tensor(CAMERA_LATERAL_AXIS, device=quat.device).expand(quat.shape[0], 3)
    lat_w = quat_apply(quat, lateral)
    return torch.nan_to_num(lat_w[:, 2] ** 2, nan=0.0)


def stall_cost(
    env,
    command_name: str = "twist",
    min_speed: float = 0.05,
    min_yaw_rate: float = 0.15,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Shortfall of the motion that was asked for, 0..1. Cost >= 0.

    For a commanded walk (|v_cmd| > min_speed): 1 - (v . v_cmd) / |v_cmd|^2,
    clipped to [0, 1]; 1 = standing still, 0 = at or above the commanded
    speed. Same for a commanded turn (|wz_cmd| > min_yaw_rate). The larger of
    the two counts. Zero when nothing (or almost nothing) is commanded, so
    standing still on a stand command stays free.
    """
    asset = env.scene[asset_cfg.name]
    cmd = env.command_manager.get_command(command_name)
    v = asset.data.root_link_lin_vel_b[:, :2]
    wz = asset.data.root_link_ang_vel_b[:, 2]
    c = cmd[:, :2]
    spd2 = (c**2).sum(dim=-1)
    lin = (1.0 - (v * c).sum(dim=-1) / spd2.clamp(min=1e-6)).clamp(0.0, 1.0)
    lin = torch.where(spd2 > min_speed**2, lin, torch.zeros_like(lin))
    yaw = (1.0 - wz * cmd[:, 2] / (cmd[:, 2] ** 2).clamp(min=1e-6)).clamp(0.0, 1.0)
    yaw = torch.where(cmd[:, 2].abs() > min_yaw_rate, yaw, torch.zeros_like(yaw))
    return torch.nan_to_num(torch.maximum(lin, yaw), nan=0.0)


_COST_FUNCS = {
    "camera_ang_rate": camera_ang_rate_cost,
    "camera_lin_acc": camera_lin_acc_cost,
    "camera_roll": camera_roll_cost,
}


# --------------------------------------------------------------------------
# Env config
# --------------------------------------------------------------------------
def make_steadycam_env_cfg(play: bool = False):
    # The C3 recipe (vendored velocity task, standard randomization), so the
    # warm-started policy sees exactly the world it was trained in.
    cfg = _vel.make_microduck_velocity_env_cfg(play=play)
    _collapse_curricula_to_final(cfg)

    twist = cfg.commands["twist"]
    twist.ranges.lin_vel_x = CINE_LIN_VEL_X
    twist.ranges.lin_vel_y = CINE_LIN_VEL_Y
    twist.ranges.ang_vel_z = CINE_ANG_VEL_Z

    if "push_robot" in cfg.events and not play:
        vr = cfg.events["push_robot"].params["velocity_range"]
        vr["x"] = CINE_PUSH_RANGE
        vr["y"] = CINE_PUSH_RANGE

    cfg.rewards["head_pose_tracking"].weight = HEAD_TRACKING_WEIGHT

    for name, func in _COST_FUNCS.items():
        cfg.rewards[name] = RewardTermCfg(func=func, weight=0.0, params={})
        final = CAMERA_WEIGHTS[name]
        stages = [
            {"step": it * _vel.NUM_STEPS_PER_ENV, "weight": final * frac}
            for it, frac in zip(RAMP_ITERS, (0.0, 1 / 3, 2 / 3, 1.0))
        ]
        cfg.curriculum[f"{name}_weight"] = CurriculumTermCfg(
            func=microduck_mdp.reward_weight,
            params={"reward_name": name, "weight_stages": stages},
        )
    return cfg


# --- v2 ---------------------------------------------------------------------
# v1 (2026-10-10) stopped walking: by snapshot 250, with the camera costs at a
# third of their final weight, it stalled on 57-99% of moving shots; from 500
# on, 98-100%. At cinematic speeds standing still loses little tracking
# reward (0.12 m/s: 87% of it), and walking also pays every gait regularizer,
# so any camera cost made standing the better deal. v2 changes the incentive,
# not just the weights:
#   * stall_cost at -2.0, full weight from iteration 0: standing still while
#     asked to move now costs 2 per second, about 3x the camera costs of the
#     unmodified C3x gait (~0.7 per second at its measured shake).
#   * linear tracking sharpened (std^2 0.1 -> 0.05): standing at 0.22 m/s
#     keeps 38% of the tracking reward instead of 62%. Angular tracking is
#     left alone: it also scores pitch/roll rates, which walking cannot avoid.
#   * camera costs ramp in over 900 iterations instead of 600.
STALL_WEIGHT = -2.0
V2_LIN_TRACK_STD = 0.05**0.5
RAMP_ITERS_V2 = (0, 300, 600, 900)


def make_steadycam_v2_env_cfg(play: bool = False):
    cfg = make_steadycam_env_cfg(play=play)
    cfg.rewards["stall"] = RewardTermCfg(func=stall_cost, weight=STALL_WEIGHT, params={})
    cfg.rewards["track_linear_velocity"].params["std"] = V2_LIN_TRACK_STD
    for name, final in CAMERA_WEIGHTS.items():
        cfg.curriculum[f"{name}_weight"].params["weight_stages"] = [
            {"step": it * _vel.NUM_STEPS_PER_ENV, "weight": final * frac}
            for it, frac in zip(RAMP_ITERS_V2, (0.0, 1 / 3, 2 / 3, 1.0))
        ]
    return cfg


def rl_cfg():
    cfg = copy.deepcopy(_vel.MicroduckRlCfg)
    cfg.experiment_name = EXPERIMENT
    cfg.run_name = EXPERIMENT
    cfg.wandb_project = "microduck_pretrain"
    cfg.max_iterations = DEFAULT_ITERATIONS
    return cfg


def rl_cfg_v2():
    cfg = rl_cfg()
    cfg.experiment_name = EXPERIMENT_V2
    cfg.run_name = EXPERIMENT_V2
    return cfg


def register() -> None:
    """Register the tasks once (safe to call again)."""
    from mjlab.tasks.registry import list_tasks, register_mjlab_task

    have = set(list_tasks())
    if TASK_ID_V2 not in have:
        register_mjlab_task(
            task_id=TASK_ID_V2,
            env_cfg=make_steadycam_v2_env_cfg(),
            play_env_cfg=make_steadycam_v2_env_cfg(play=True),
            rl_cfg=rl_cfg_v2(),
            runner_cls=_md_tasks.MicroduckOnPolicyRunner,
        )
    if TASK_ID in have:
        return
    register_mjlab_task(
        task_id=TASK_ID,
        env_cfg=make_steadycam_env_cfg(),
        play_env_cfg=make_steadycam_env_cfg(play=True),
        rl_cfg=rl_cfg(),
        runner_cls=_md_tasks.MicroduckOnPolicyRunner,
    )


# Self-register when the module finishes loading. `import mjlab` loads every
# `mjlab.tasks` plugin, including microduck_pretrain.tasks, which imports this
# module; if this module was imported first it is still half-loaded at that
# moment, so registration has to happen here, at the end, not from tasks.py.
register()
