"""Reward-weight screen: the design is a proper two-level screen, the weight
multipliers reach the task, and the analysis recovers planted effects."""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

from microduck_pretrain import reward_screen as RS

REPO = Path(__file__).resolve().parents[1]


def _driver():
    spec = importlib.util.spec_from_file_location("reward_screen_driver", REPO / "scripts" / "reward_screen.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_design_is_balanced_and_orthogonal():
    x = RS.design_matrix()
    assert x.shape == (16, len(RS.FACTORS)) == (16, 15)
    assert (x.sum(axis=0) == 0).all()                      # 8 high, 8 low per weight
    assert np.array_equal(x.T @ x, 16 * np.eye(15))         # columns orthogonal
    rows = RS.runs()
    assert len(rows) == 16 + RS.N_CENTER
    assert len({r["run"] for r in rows}) == len(rows)
    centers = [r for r in rows if r["kind"] == "center"]
    assert [r["seed"] for r in centers] == [1, 2, 3, 4]
    assert all(set(r["scales"].values()) == {1.0} for r in centers)
    for r in rows:
        if r["kind"] == "screen":
            assert set(r["scales"].values()) <= {RS.LOW, RS.HIGH}
            assert all(r["scales"][f] == (RS.HIGH if r["levels"][f] > 0 else RS.LOW) for f in RS.FACTORS)
    assert rows[0]["kind"] == rows[-1]["kind"] == "center"   # centers spread through the order


def test_scales_from_env(monkeypatch):
    monkeypatch.delenv(RS.SCALES_ENV, raising=False)
    assert RS.scales_from_env() == {}
    monkeypatch.setenv(RS.SCALES_ENV, '{"air_time": 2.0, "foot_slip": 0.5}')
    assert RS.scales_from_env() == {"air_time": 2.0, "foot_slip": 0.5}
    monkeypatch.setenv(RS.SCALES_ENV, '{"not_a_term": 2.0}')
    with pytest.raises(ValueError):
        RS.scales_from_env()


def test_analysis_recovers_planted_effects():
    drv = _driver()
    rows = RS.runs()
    rng = np.random.default_rng(0)
    resp = {}
    for r in rows:
        lv = r["levels"]
        base = {k: 0.1 for k, _, _ in drv.RESPONSES}
        base["nom_speed"] += -0.02 * lv["track_linear_velocity"] + 0.01 * lv["air_time"]
        base["careful_stall"] = 30.0 + 8.0 * lv["head_pose_bias"]
        resp[r["run"]] = {k: v + rng.normal(0, 0.002 if k != "careful_stall" else 0.5) for k, v in base.items()}
    E = drv.effects(rows, resp)
    nom = E["nom_speed"]
    assert nom["effects"]["track_linear_velocity"] == pytest.approx(-0.04, abs=0.006)
    assert nom["effects"]["air_time"] == pytest.approx(0.02, abs=0.006)
    assert drv._mark(nom["effects"]["track_linear_velocity"], nom) == "**"
    assert drv._mark(nom["effects"]["pose"], nom) == ""
    assert E["careful_stall"]["effects"]["head_pose_bias"] == pytest.approx(16.0, abs=1.5)
    assert sum(drv._mark(v, nom) != "" for v in nom["effects"].values()) <= 3


def test_analysis_tolerates_missing_and_nan():
    drv = _driver()
    rows = RS.runs()
    resp = {r["run"]: {k: float("nan") for k, _, _ in drv.RESPONSES} for r in rows[:5]}
    E = drv.effects(rows, resp)
    assert all(np.isnan(v) for v in E["nom_speed"]["effects"].values())


@pytest.mark.slow
def test_screen_task_scales_c3x_weights(monkeypatch):
    from mjlab.tasks.registry import list_tasks

    from microduck_pretrain.followups import make_c3x_env_cfg

    assert RS.SCREEN_TASK in set(list_tasks())
    base = make_c3x_env_cfg()
    active = {k for k, v in base.rewards.items() if v.weight != 0}
    assert active == set(RS.FACTORS)                        # the design covers every active term
    monkeypatch.setenv(RS.SCALES_ENV, '{"air_time": 2.0, "foot_slip": 0.5}')
    cfg = RS.make_screen_env_cfg()
    assert cfg.rewards["air_time"].weight == pytest.approx(2.0 * base.rewards["air_time"].weight)
    assert cfg.rewards["foot_slip"].weight == pytest.approx(0.5 * base.rewards["foot_slip"].weight)
    assert cfg.rewards["upright"].weight == base.rewards["upright"].weight
    assert make_c3x_env_cfg().rewards["air_time"].weight == base.rewards["air_time"].weight  # C3x untouched
    monkeypatch.delenv(RS.SCALES_ENV)
    assert {k: v.weight for k, v in RS.make_screen_env_cfg().rewards.items()} == \
        {k: v.weight for k, v in base.rewards.items()}
