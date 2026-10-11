"""Run the CPU evaluator on a deterministic do-nothing policy."""

import math

import pytest

from microduck_pretrain.evaluate import Scenario, Segment, run_scenario


@pytest.mark.slow
@pytest.mark.parametrize("scene,rough", [("flat", 0.0), ("backlash", 6.0)])
def test_zero_policy_does_not_walk(zero_policy, scene, rough):
    sc = Scenario(
        name="t", scene=scene, rough_height_mm=rough, payload_kg=0.05, duration_s=4.0,
        commands=[Segment(vx=0.3, seconds=4.0)],
    )
    r = run_scenario(sc, zero_policy, seed=0)
    # Holding the default pose never produces walking: it either stands still
    # (stalled) or tips over (falls) - both must register in the metrics.
    assert r.falls > 0 or r.stall_fraction > 0.9
    assert math.isfinite(r.lin_vel_err_mps) and r.lin_vel_err_mps > 0.2
    assert 0.0 <= r.upright_fraction <= 1.0


@pytest.mark.slow
def test_per_command_breakdown(zero_policy):
    from dataclasses import asdict

    from microduck_pretrain.evaluate import per_command_summary

    sc = Scenario(name="t", duration_s=6.0,
                  commands=[Segment(vx=0.1, seconds=3.0), Segment(seconds=1.0), Segment(wz=0.3, seconds=2.0)])
    r = run_scenario(sc, zero_policy, seed=0)
    assert [(c["vx"], c["wz"]) for c in r.per_command] == [(0.1, 0.0), (0.0, 0.0), (0.0, 0.3)]
    assert r.per_command[1]["moving"] == 0                    # stand segment is not judged
    assert sum(c["stalled"] for c in r.per_command) <= sum(c["moving"] for c in r.per_command)
    rows = per_command_summary([asdict(r), asdict(r)])
    assert all(c["index"] != 1 for c in rows)                 # stands left out
    assert all(0.0 <= c["stall_fraction"] <= 1.0 for c in rows)


def test_per_command_summary_pools_runs():
    from microduck_pretrain.evaluate import per_command_summary

    a = {"per_command": [{"vx": 0.1, "vy": 0.0, "wz": 0.0, "moving": 10, "stalled": 10, "achieved_sum": 0.0}]}
    b = {"per_command": [{"vx": 0.1, "vy": 0.0, "wz": 0.0, "moving": 30, "stalled": 0, "achieved_sum": 30.0}]}
    (row,) = per_command_summary([a, b])
    assert row["stall_fraction"] == pytest.approx(0.25)
    assert row["achieved_fraction"] == pytest.approx(0.75)
    assert per_command_summary([{"per_command": []}, {}]) == []
