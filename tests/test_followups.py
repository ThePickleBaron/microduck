"""C5 (site fine-tune) and C3x (control): the progress-based terrain rule
promotes and demotes the right robots, and the new tasks leave C1-C4 alone."""

from types import SimpleNamespace

import pytest
import torch


class _Terrain:
    def __init__(self, n):
        self.cfg = SimpleNamespace(terrain_generator=object())
        self.terrain_levels = torch.zeros(n, dtype=torch.long)
        self.calls = []

    def update_env_origins(self, env_ids, move_up, move_down):
        self.calls.append((env_ids.clone(), move_up.clone(), move_down.clone()))
        self.terrain_levels[env_ids] += move_up.long() - move_down.long()


class _Scene:
    def __init__(self, n, robot):
        self.terrain = _Terrain(n)
        self.env_origins = torch.zeros(n, 3)
        self._robot = robot

    def __getitem__(self, name):
        return self._robot


def _fake_env(actual_xy, expected_xy):
    n = len(actual_xy)
    pos = torch.zeros(n, 3)
    pos[:, :2] = torch.tensor(actual_xy, dtype=torch.float)
    robot = SimpleNamespace(data=SimpleNamespace(root_link_pos_w=pos))
    env = SimpleNamespace(num_envs=n, device="cpu", scene=_Scene(n, robot))
    env._md_expected_disp = torch.tensor(expected_xy, dtype=torch.float)
    return env


def test_progress_rule_promotes_demotes_and_skips():
    from microduck_pretrain import followups as f

    env = _fake_env(
        actual_xy=[(2.5, 0.0), (0.5, 0.0), (1.4, 0.0), (0.0, 0.0), (0.0, 2.8)],
        expected_xy=[(3.0, 0.0), (3.0, 0.0), (3.0, 0.0), (0.4, 0.0), (3.0, 0.0)],
    )
    ids = torch.arange(5)
    f.terrain_levels_progress(env, ids)
    _, up, down = env.scene.terrain.calls[0]
    # 0: 83% of the way -> up; 1: 17% -> down; 2: 47% -> stays;
    # 3: asked for 0.4 m (< 1 m) -> not judged; 4: went sideways (0 along) -> down
    assert up.tolist() == [True, False, False, False, False]
    assert down.tolist() == [False, True, False, False, True]
    assert torch.all(env._md_expected_disp == 0), "buffer must clear at reset"


def test_old_rule_would_almost_never_promote():
    """mjlab's rule needs half a tile (4 m) in one 20 s episode."""
    from microduck_pretrain.tasks import make_site_env_cfg

    tile = make_site_env_cfg().scene.terrain.terrain_generator.size[0]
    assert tile / 2 >= 4.0  # 0.2 m/s typical x 20 s = 4.0 m: not strictly greater


@pytest.mark.slow
class TestFollowupCfgs:
    def test_registered(self):
        from mjlab.tasks.registry import list_tasks

        from microduck_pretrain import followups as f
        from microduck_pretrain.conditions import TRAINABLE

        have = set(list_tasks())
        assert {f.C5_TASK, f.C3X_TASK} <= have
        assert all(c.task_id in have for c in TRAINABLE)

    def test_c5_is_c4_world_with_new_promotion(self):
        from microduck_pretrain import followups as f
        from microduck_pretrain import tasks

        c4, c5 = tasks.make_site_env_cfg(), f.make_c5_env_cfg()
        assert c5.events["payload"].params == c4.events["payload"].params
        assert c5.events["foot_friction"].params["ranges"] == c4.events["foot_friction"].params["ranges"]
        assert type(c5.scene.entities["robot"].articulation.actuators[0]) is type(
            c4.scene.entities["robot"].articulation.actuators[0]
        )
        assert c5.curriculum["terrain_levels"].func is f.terrain_levels_progress
        assert c4.curriculum["terrain_levels"].func is not f.terrain_levels_progress  # C4 untouched
        assert c5.rewards["action_rate_l2"].weight == -1.0  # pinned at C3's final stage

    def test_c3x_is_c3_world(self):
        from microduck_pretrain import followups as f
        from microduck_pretrain import tasks

        c3, c3x = tasks.make_standard_env_cfg(), f.make_c3x_env_cfg()
        assert set(c3x.rewards) == set(c3.rewards)
        assert c3x.commands["twist"].ranges == c3.commands["twist"].ranges
        assert list(c3x.observations["actor"].terms) == list(c3.observations["actor"].terms)
