"""Fast checks on the experiment definition (no simulator imports)."""

from microduck_pretrain import conditions as c
from microduck_pretrain.train import build_command


def test_condition_keys_and_tasks_unique():
    keys = [x.key for x in c.CONDITIONS]
    assert keys == ["c1", "c2", "c3", "c4"]
    tasks = [x.task_id for x in c.TRAINABLE]
    assert len(tasks) == len(set(tasks)) == 3
    assert c.VENDOR_DEFAULT.task_id is None


def test_by_key_is_case_insensitive():
    assert c.by_key("C4") is c.SITE_SPECIFIC


def test_train_command_pins_the_budget():
    cmd = build_command("c3", seed=2, num_envs=1024, iterations=500, logger="tensorboard", extra=["--hf-jobs"])
    assert cmd[:2] == ["train", c.STANDARD.task_id]
    flags = dict(zip(cmd[2:-1:2], cmd[3:-1:2]))
    assert flags["--env.scene.num-envs"] == "1024"
    assert flags["--agent.max-iterations"] == "500"
    assert flags["--agent.seed"] == "2"
    assert flags["--agent.run-name"] == "c3_seed2"
    assert cmd[-1] == "--hf-jobs"


def test_site_ranges_are_wider_than_standard_recipe():
    lo, hi = c.SITE_FOOT_FRICTION
    assert lo < 0.7 and hi > 1.3          # standard recipe trains on (0.7, 1.3)
    assert c.SITE_PAYLOAD_KG[0] == 0.0 and c.SITE_PAYLOAD_KG[1] > 0
