"""Register the trainable experiment conditions as mjlab tasks.

Loaded automatically through the `mjlab.tasks` entry point (pyproject.toml)
whenever `train`, `play` or `list-envs` runs, right next to the vendored
Microduck tasks. Each condition is built FROM the vendored velocity task, so
the reward recipe, observation layout (61-D) and ONNX export stay identical to
the official policies; only the randomization and environment differ. See
conditions.py for what each condition means.
"""

from __future__ import annotations

import contextlib
import copy
import dataclasses
import warnings

# The payload is deliberately a point mass (see make_site_env_cfg), which is
# exactly the case mjlab's body_mass warning says is appropriate.
warnings.filterwarnings("ignore", message=r"dr\.body_mass only randomizes mass")

# Importing the vendored package registers its own tasks first (and handles a
# `train ... --hf-jobs` submission before anything heavy is built).
import mjlab_microduck.tasks as _md_tasks
from mjlab.tasks.registry import register_mjlab_task
from mjlab_microduck.robot import microduck_constants as _mc
from mjlab_microduck.tasks import microduck_velocity_env_cfg as _vel

from microduck_pretrain.conditions import (
    IDEAL_VIN,
    IDEALIZED,
    SITE_FOOT_FRICTION,
    SITE_PAYLOAD_KG,
    SITE_SPECIFIC,
    STANDARD,
)

# Every domain-randomization toggle at the top of the vendored velocity cfg.
# Read at registration time so a new upstream toggle is caught by the test
# suite (tests/test_tasks.py) rather than silently left on.
DR_FLAGS: tuple[str, ...] = tuple(
    name
    for name in dir(_vel)
    if name.startswith("ENABLE_") and name != "ENABLE_SYMMETRY"
)


@contextlib.contextmanager
def _patched(module, **values):
    """Temporarily override module-level constants while a cfg is built."""
    saved = {k: getattr(module, k) for k in values}
    try:
        for k, v in values.items():
            setattr(module, k, v)
        yield
    finally:
        for k, v in saved.items():
            setattr(module, k, v)


def _rl_cfg(experiment_name: str):
    """Copy of the vendored PPO config with its own log directory."""
    cfg = copy.deepcopy(_vel.MicroduckRlCfg)
    cfg.experiment_name = experiment_name
    cfg.run_name = experiment_name
    cfg.wandb_project = "microduck_pretrain"
    return cfg


# --------------------------------------------------------------------------
# C2 - idealized simulation
# --------------------------------------------------------------------------
def _ideal_robot_cfg():
    """Walk robot with an ideal BAM actuator: fixed voltage, no sag, no delay."""
    kwargs = dict(_mc._BAM_ACTUATOR_KWARGS)
    kwargs.update(
        vin=IDEAL_VIN,
        vin_range=None,
        vin_drop_gain_range=None,
        vin_min=None,
        delay_min_lag=0,
        delay_max_lag=0,
    )
    actuator = type(_mc.actuators)(**kwargs)
    robot = _mc.MICRODUCK_WALK_ROBOT_CFG
    return dataclasses.replace(
        robot,
        articulation=dataclasses.replace(robot.articulation, actuators=(actuator,)),
    )


def make_idealized_env_cfg(play: bool = False):
    with _patched(_vel, **{flag: False for flag in DR_FLAGS}):
        cfg = _vel.make_microduck_velocity_env_cfg(play=play)
    cfg.scene.entities = {"robot": _ideal_robot_cfg()}
    cfg.events["foot_friction"].params["ranges"] = (1.0, 1.0)
    # Two randomizations come from mjlab's base velocity cfg rather than the
    # ENABLE_* toggles: random pushes and a +/-2.5 cm trunk CoM offset.
    for name in ("push_robot", "base_com"):
        cfg.events.pop(name, None)
    for term in cfg.observations["actor"].terms.values():
        term.noise = None
    return cfg


# --------------------------------------------------------------------------
# C3 - standard randomization (the vendored recipe, unchanged)
# --------------------------------------------------------------------------
def make_standard_env_cfg(play: bool = False):
    return _vel.make_microduck_velocity_env_cfg(play=play)


# --------------------------------------------------------------------------
# C4 - site-specific pre-training
# --------------------------------------------------------------------------
def make_site_env_cfg(play: bool = False):
    cfg = _vel.make_microduck_velocity_env_cfg(play=play, rough=True)
    cfg = _md_tasks.make_backlash_variant(cfg, _mc.MICRODUCK_WALK_BACKLASH_ROBOT_CFG)
    cfg.events["foot_friction"].params["ranges"] = SITE_FOOT_FRICTION
    # A payload is a point mass bolted to the trunk: body_mass with
    # operation="add" is the physically right model (no inertia change).
    cfg.events["payload"] = _vel.EventTermCfg(
        func=_vel.dr.body_mass,
        mode="startup",
        params={
            "asset_cfg": _vel.SceneEntityCfg("robot", body_names=("trunk_base",)),
            "ranges": SITE_PAYLOAD_KG,
            "operation": "add",
        },
    )
    return cfg


_FACTORIES = {
    IDEALIZED.task_id: (make_idealized_env_cfg, "c2_idealized"),
    STANDARD.task_id: (make_standard_env_cfg, "c3_standard"),
    SITE_SPECIFIC.task_id: (make_site_env_cfg, "c4_site"),
}

for _task_id, (_make, _exp) in _FACTORIES.items():
    register_mjlab_task(
        task_id=_task_id,
        env_cfg=_make(),
        play_env_cfg=_make(play=True),
        rl_cfg=_rl_cfg(_exp),
        runner_cls=_md_tasks.MicroduckOnPolicyRunner,
    )
