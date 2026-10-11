"""Reward surface on C6v2: a 3x3 factorial with replicates, multipliers
that reach the task, and a fit that recovers a planted quadratic."""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

from microduck_pretrain import reward_surface as RF

REPO = Path(__file__).resolve().parents[1]


def _driver():
    spec = importlib.util.spec_from_file_location("reward_surface_driver", REPO / "scripts" / "reward_surface.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_design_is_full_factorial_with_replicates():
    rows = RF.runs()
    assert len(rows) == 18 and len({r["run"] for r in rows}) == 18
    for seed in RF.SEEDS:
        cells = {(r["x"]["linear"], r["x"]["turning"]) for r in rows if r["seed"] == seed}
        assert cells == {(a, b) for a in (-1, 0, 1) for b in (-1, 0, 1)}
    for r in rows:
        assert r["parent_seed"] == r["seed"]
        for t in RF.FACTORS["linear"]:
            assert r["scales"][t] == 2.0 ** r["x"]["linear"]
        for t in RF.FACTORS["turning"]:
            assert r["scales"][t] == 2.0 ** r["x"]["turning"]
    assert [r["run"] for r in rows] == [r["run"] for r in RF.runs()]          # fixed order
    assert [r["seed"] for r in rows] != sorted(r["seed"] for r in rows)        # but not by seed


def test_scales_env_is_separate_from_the_screen(monkeypatch):
    from microduck_pretrain import reward_screen as RS

    assert RF.SCALES_ENV != RS.SCALES_ENV
    monkeypatch.setenv(RF.SCALES_ENV, '{"track_lin_relative": 2.0}')
    assert RF.scales_from_env() == {"track_lin_relative": 2.0}
    monkeypatch.setenv(RF.SCALES_ENV, '{"air_time": 2.0}')
    with pytest.raises(ValueError):
        RF.scales_from_env()


def test_fit_recovers_a_planted_quadratic():
    drv = _driver()
    rng = np.random.default_rng(0)
    pts = [(a, b, 0.03 - 0.01 * a + 0.008 * b + 0.004 * a * a + 0.002 * a * b + rng.normal(0, 0.0005))
           for a in (-1, 0, 1) for b in (-1, 0, 1) for _ in range(2)]
    f = drv.fit(pts)
    assert f["ok"] and f["df"] == 12
    assert f["beta"] == pytest.approx([0.03, -0.01, 0.008, 0.002, 0.004, 0.0], abs=0.0015)
    assert f["r2"] > 0.95
    assert drv.fit(pts[:5])["ok"] is False


def test_operating_point_balances_speed_and_turning():
    drv = _driver()
    pts = lambda fn: [(a, b, fn(a, b)) for a in (-1, 0, 1) for b in (-1, 0, 1) for _ in range(2)]  # noqa: E731
    fits = {
        "ho_speed": drv.fit(pts(lambda a, b: 0.033 - 0.01 * a + 0.01 * b + 1e-4 * ((a + b) % 2))),
        "ho_yaw": drv.fit(pts(lambda a, b: 0.11 + 0.03 * a - 0.03 * b + 1e-4 * ((a * b) % 2))),
        "careful_stall": drv.fit(pts(lambda a, b: 1.0 + 0.1 * a + 1e-3 * (a % 2))),
        "ho_falls": drv.fit(pts(lambda a, b: 0.3 + 0.01 * b + 1e-4 * (b % 2))),
    }
    op = drv.operating_point(fits)
    assert op is not None
    assert op["b"] - op["a"] > 0.3                       # yaw is the binding target: shift toward turning
    assert op["ho_speed"] / drv.SPEED_TARGET == pytest.approx(op["ho_yaw"] / drv.YAW_TARGET, rel=0.05)  # balanced
    assert op["J"] < op["J_center"]


@pytest.mark.slow
def test_surface_task_scales_c6v2_pairs(monkeypatch):
    from mjlab.tasks.registry import list_tasks

    from microduck_pretrain.followups import make_c6v2_env_cfg

    assert RF.SURFACE_TASK in set(list_tasks())
    base = make_c6v2_env_cfg()
    monkeypatch.setenv(RF.SCALES_ENV, '{"track_linear_velocity": 2.0, "track_lin_relative": 2.0, '
                                      '"track_angular_velocity": 0.5, "track_yaw_relative": 0.5}')
    cfg = RF.make_surface_env_cfg()
    for t, m in [("track_linear_velocity", 2.0), ("track_lin_relative", 2.0),
                 ("track_angular_velocity", 0.5), ("track_yaw_relative", 0.5), ("air_time", 1.0)]:
        assert cfg.rewards[t].weight == pytest.approx(m * base.rewards[t].weight)
    assert type(cfg.commands["twist"]).__name__ == "CarefulVelocityCommandCfg"   # still C6v2's commands
