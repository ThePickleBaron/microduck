"""Steady-cam side project: registers without disturbing C1-C4, keeps the
shared observation contract, and its costs have the right sign and frame."""

import sys
from pathlib import Path

import mujoco
import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
WALK_SCENE = REPO / "third_party/microduck_rl/src/mjlab_microduck/robot/microduck/scene_walk.xml"


def test_camera_lateral_axis_is_site_y():
    """The roll cost reads site +Y as the image's left-right axis. With every
    joint at zero the head_camera site must be world-aligned (+Y = left)."""
    from microduck_pretrain import steadycam

    model = mujoco.MjModel.from_xml_path(str(WALK_SCENE))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, steadycam.CAMERA_SITE)
    assert sid >= 0
    lateral_w = data.site_xmat[sid].reshape(3, 3) @ np.array(steadycam.CAMERA_LATERAL_AXIS)
    assert abs(lateral_w[1]) > 0.99 and abs(lateral_w[2]) < 1e-6


@pytest.fixture(scope="module")
def cfgs():
    from microduck_pretrain import steadycam, tasks

    return {"c3": tasks.make_standard_env_cfg(), "sc": steadycam.make_steadycam_env_cfg()}


@pytest.mark.slow
class TestSteadycamCfg:
    def test_registered_next_to_the_conditions(self):
        from mjlab.tasks.registry import list_tasks

        from microduck_pretrain import steadycam
        from microduck_pretrain.conditions import TRAINABLE

        have = set(list_tasks())
        assert steadycam.TASK_ID in have
        assert all(c.task_id in have for c in TRAINABLE)

    def test_c3_is_untouched(self, cfgs):
        """Building the steady-cam cfg must not leak into the experiment."""
        from microduck_pretrain import tasks

        fresh = tasks.make_standard_env_cfg()
        assert set(fresh.rewards) == set(cfgs["c3"].rewards)
        assert not any(k.startswith("camera_") for k in fresh.rewards)
        assert fresh.rewards["head_pose_tracking"].weight == 2.0
        assert fresh.commands["twist"].ranges.lin_vel_x == (-0.4, 0.4)
        assert len(fresh.curriculum["action_rate_weight"].params["weight_stages"]) > 1

    def test_observation_contract_unchanged(self, cfgs):
        for grp in ("actor", "critic"):
            assert list(cfgs["sc"].observations[grp].terms) == list(cfgs["c3"].observations[grp].terms)

    def test_costs_negative_and_ramped_from_zero(self, cfgs):
        from microduck_pretrain import steadycam

        sc = cfgs["sc"]
        for name, final in steadycam.CAMERA_WEIGHTS.items():
            assert final < 0, name  # costs return >= 0, so weights must be negative
            assert sc.rewards[name].weight == 0.0
            stages = sc.curriculum[f"{name}_weight"].params["weight_stages"]
            assert stages[0]["weight"] == 0 and stages[-1]["weight"] == final

    def test_inherited_curricula_pinned_at_final(self, cfgs):
        sc = cfgs["sc"]
        assert sc.rewards["action_rate_l2"].weight == -1.0
        assert sc.curriculum["standing_envs"].params["standing_stages"] == [
            {"step": 0, "rel_standing_envs": 0.25}
        ]
        assert sc.rewards["head_pose_tracking"].weight < cfgs["c3"].rewards["head_pose_tracking"].weight


def test_eval_runs_and_reports_finite_metrics(zero_policy, tmp_path):
    """The shot evaluator runs end to end and writes finite camera metrics.
    (A zero-action policy may tip over, as in test_evaluate.py; falls are
    counted and the camera metrics still cover the upright time.)"""
    sys.path.insert(0, str(REPO / "scripts"))
    import eval_steadycam

    rc = eval_steadycam.main([
        "--policy", str(zero_policy), "--label", "zero", "--only", "locked_off",
        "--seeds", "1", "--duration", "5", "--out", str(tmp_path),
    ])
    assert rc == 0
    import json

    rep = json.loads((tmp_path / "steadycam__zero.json").read_text())
    shot = rep["shots"]["locked_off"]
    assert shot["falls"] >= 0
    for key in ("ang_rate_dps", "ang_rate_p95", "lin_acc_mps2", "roll_deg"):
        assert np.isfinite(shot[key]) and shot[key] >= 0, key


def _stall_env(v_body, wz, cmd):
    from types import SimpleNamespace

    import torch

    v = torch.tensor(v_body, dtype=torch.float)
    data = SimpleNamespace(
        root_link_lin_vel_b=torch.cat([v, torch.zeros(len(v), 1)], dim=1),
        root_link_ang_vel_b=torch.stack([torch.zeros(len(wz)), torch.zeros(len(wz)), torch.tensor(wz)], dim=1),
    )
    cmds = torch.tensor(cmd, dtype=torch.float)
    return SimpleNamespace(
        scene={"robot": SimpleNamespace(data=data)},
        command_manager=SimpleNamespace(get_command=lambda name: cmds),
    )


def test_stall_cost_prices_standing_still_only_when_asked_to_move():
    from microduck_pretrain import steadycam

    env = _stall_env(
        v_body=[(0.0, 0.0), (0.20, 0.0), (0.10, 0.0), (0.0, 0.0), (0.0, 0.0), (0.0, 0.0), (0.0, 0.0)],
        wz=[0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0],
        cmd=[(0.2, 0, 0), (0.2, 0, 0), (0.2, 0, 0), (0.0, 0, 0), (0.02, 0, 0), (0, 0, 0.5), (0, 0, 0.5)],
    )
    cost = steadycam.stall_cost(env).tolist()
    # standing on a walk command: 1; at speed: 0; half speed: 0.5; standing on a stand
    # command or a tiny one: free; turning as asked: 0; not turning: 1
    assert cost == pytest.approx([1.0, 0.0, 0.5, 0.0, 0.0, 0.0, 1.0])


@pytest.mark.slow
def test_v2_adds_stall_and_sharper_tracking_and_leaves_v1_alone():
    from mjlab.tasks.registry import list_tasks

    from microduck_pretrain import steadycam

    assert steadycam.TASK_ID_V2 in set(list_tasks())
    v1, v2 = steadycam.make_steadycam_env_cfg(), steadycam.make_steadycam_v2_env_cfg()
    assert "stall" not in v1.rewards and v2.rewards["stall"].weight == steadycam.STALL_WEIGHT
    assert v2.rewards["track_linear_velocity"].params["std"] < v1.rewards["track_linear_velocity"].params["std"]
    assert v2.rewards["track_angular_velocity"].params["std"] == v1.rewards["track_angular_velocity"].params["std"]
    last = v2.curriculum["camera_roll_weight"].params["weight_stages"][-1]
    assert last["weight"] == steadycam.CAMERA_WEIGHTS["camera_roll"]
