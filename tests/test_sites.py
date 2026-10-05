"""The site files parse and the held-out site really is held out."""

from microduck_pretrain import conditions as c
from microduck_pretrain.evaluate import SCENES, load_site

from conftest import REPO


def test_sites_parse():
    for name in ("nominal", "held_out"):
        meta, scenarios = load_site(REPO / "sites" / f"{name}.toml")
        assert meta["name"] == name
        assert scenarios and all(s.commands for s in scenarios)
        assert all(s.scene in SCENES for s in scenarios)


def test_held_out_scenarios_leave_the_training_ranges():
    _, scenarios = load_site(REPO / "sites" / "held_out.toml")
    by_name = {s.name: s for s in scenarios}
    assert by_name["slippery_floor"].foot_friction < c.SITE_FOOT_FRICTION[0]
    assert by_name["heavy_payload"].payload_kg > c.SITE_PAYLOAD_KG[1]
    assert by_name["low_battery"].vin < 6.5            # training samples 6.5-8.2 V
    assert by_name["worn_gears"].scene == "backlash"
    combined = by_name["site_combined"]
    assert combined.rough_height_mm > 0 and combined.payload_kg > 0 and combined.scene == "backlash"


def test_nominal_stays_inside_training_ranges():
    _, (nominal,) = load_site(REPO / "sites" / "nominal.toml")
    assert 0.7 <= nominal.foot_friction <= 1.3
    assert 6.5 <= nominal.vin <= 8.2
    assert nominal.payload_kg == 0 and nominal.rough_height_mm == 0 and nominal.scene == "flat"
