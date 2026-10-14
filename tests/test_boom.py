"""Boom model: beam numbers, the boom attaches without disturbing the
robot's joints, and it starts at its static sag."""

import math

import mujoco
import numpy as np
import pytest

from microduck_pretrain import evaluate as ev


def test_boom_params_match_beam_theory():
    sc = ev.Scenario(name="b", boom_length_m=0.91, boom_rod_mm=3.0, boom_tip_g=15.0, boom_e_gpa=110.0)
    bp = ev.boom_params(sc)
    EI = 110e9 * math.pi * 0.0015**4 / 4
    assert bp["EI"] == pytest.approx(EI)
    assert bp["m_rod"] == pytest.approx(1550 * math.pi * 0.0015**2 * 0.91)
    k_tip = 3 * EI / 0.91**3
    assert bp["f1_hz"] == pytest.approx(math.sqrt(k_tip / (0.015 + 0.236 * bp["m_rod"])) / (2 * math.pi))
    assert 1.4 < bp["f1_hz"] < 1.8 and 90 < bp["static_tip_mm_horizontal"] < 120
    assert bp["I_root"] * (2 * math.pi * bp["f1_hz"]) ** 2 == pytest.approx(bp["k_theta"])
    # a tube is stiffer for its mass
    tube = ev.boom_params(ev.Scenario(name="t", boom_length_m=0.91, boom_rod_mm=4.0, boom_bore_mm=2.0))
    assert tube["f1_hz"] > bp["f1_hz"]


@pytest.mark.slow
@pytest.mark.parametrize("direction", ["forward", "side", "up"])
def test_boom_attaches_after_robot_joints_and_settles(direction):
    plain_m, _, _ = ev.build_sim(ev.Scenario(name="n"), np.random.default_rng(0))
    sc = ev.Scenario(name="b", boom_length_m=0.91, boom_dir=direction)
    m, d, _ = ev.build_sim(sc, np.random.default_rng(0))
    assert m.nq == plain_m.nq + 2 and m.nu == plain_m.nu
    for i in range(plain_m.njnt):                                  # robot joints keep their addresses
        assert m.jnt_qposadr[i] == plain_m.jnt_qposadr[i]
    assert m.body_subtreemass[1] == pytest.approx(plain_m.body_subtreemass[1] + 0.015 + ev.boom_params(sc)["m_rod"])
    jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "trunk_base_freejoint")
    d.qpos[m.jnt_qposadr[jid] + 2] = ev.SPAWN_HEIGHT_M
    static = ev.settle_boom(m, d)
    bp = ev.boom_params(sc)
    if direction == "up":
        assert abs(static[0]) < 1e-3 and abs(static[1]) < 1e-3
    else:
        assert static[0] == pytest.approx(bp["moment_horizontal_nm"] / bp["k_theta"], rel=0.15)  # sags down
    rigid, _, _ = ev.build_sim(ev.Scenario(name="r", boom_length_m=0.91, boom_dir=direction, boom_rigid=True),
                               np.random.default_rng(0))
    assert rigid.nq == plain_m.nq


def test_stress_cache_keys_ignore_boom_fields_without_a_boom(tmp_path):
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import stress_tests as S

    pol = tmp_path / "p.onnx"
    pol.write_bytes(b"x")
    a = S._key(pol, ev.Scenario(name="n"), 0)
    assert a == S._key(pol, ev.Scenario(name="n", boom_tip_g=99.0), 0)          # no boom: boom fields ignored
    assert a != S._key(pol, ev.Scenario(name="n", boom_length_m=0.5), 0)
