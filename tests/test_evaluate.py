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
