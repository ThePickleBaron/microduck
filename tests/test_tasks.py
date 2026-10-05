"""The condition tasks register and differ from each other the way
conditions.py says. Building the configs imports mjlab/warp (~1 min)."""

import pytest

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def cfgs():
    from microduck_pretrain import tasks

    return {
        "c2": tasks.make_idealized_env_cfg(),
        "c3": tasks.make_standard_env_cfg(),
        "c4": tasks.make_site_env_cfg(),
        "flags": tasks.DR_FLAGS,
    }


def test_tasks_registered():
    from mjlab.tasks.registry import list_tasks

    from microduck_pretrain.conditions import TRAINABLE

    have = set(list_tasks())
    assert all(c.task_id in have for c in TRAINABLE)


def test_idealized_has_no_randomization(cfgs):
    c2, c3 = cfgs["c2"], cfgs["c3"]
    dr_names = {"push_robot", "base_com", "encoder_bias"}
    dr_events = {k for k in c3.events if k.startswith("randomize_") or k in dr_names}
    assert dr_events, "standard recipe should randomize something"
    assert not (dr_events & set(c2.events)), set(c2.events)
    assert c2.events["foot_friction"].params["ranges"] == (1.0, 1.0)
    act = c2.scene.entities["robot"].articulation.actuators[0]
    assert act.vin_range is None and act.vin_drop_gain_range is None
    assert act.delay_max_lag == 0
    assert all(t.noise is None for t in c2.observations["actor"].terms.values())


def test_flags_cover_every_toggle(cfgs):
    assert "ENABLE_VELOCITY_PUSHES" in cfgs["flags"]
    assert "ENABLE_SYMMETRY" not in cfgs["flags"]


def test_site_specific_adds_payload_terrain_and_backlash(cfgs):
    from microduck_pretrain.conditions import SITE_FOOT_FRICTION, SITE_PAYLOAD_KG

    c4 = cfgs["c4"]
    assert c4.events["payload"].params["ranges"] == SITE_PAYLOAD_KG
    assert c4.events["payload"].params["operation"] == "add"
    assert c4.events["foot_friction"].params["ranges"] == SITE_FOOT_FRICTION
    assert c4.scene.terrain is not None and c4.scene.terrain.terrain_generator is not None
    assert "backlash" in type(c4.scene.entities["robot"].articulation.actuators[0]).__name__.lower()


def test_observation_layout_matches_vendor(cfgs):
    # Same 61-D actor obs as the shipped policies, so ONNX files are interchangeable.
    names = [list(cfgs[k].observations["actor"].terms) for k in ("c2", "c3", "c4")]
    assert names[0] == names[1] == names[2]
