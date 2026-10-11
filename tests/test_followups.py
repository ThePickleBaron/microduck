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


def _c6_env(v_xy, wz, cmd, steps_in=5):
    from types import SimpleNamespace

    data = SimpleNamespace(
        root_link_lin_vel_b=torch.cat([torch.tensor(v_xy, dtype=torch.float), torch.zeros(len(v_xy), 1)], dim=1),
        root_link_ang_vel_b=torch.stack([torch.zeros(len(wz)), torch.zeros(len(wz)), torch.tensor(wz)], dim=1),
    )
    cmds = torch.tensor(cmd, dtype=torch.float)
    return SimpleNamespace(
        scene={"robot": SimpleNamespace(data=data)},
        command_manager=SimpleNamespace(get_command=lambda name: cmds),
        step_dt=0.02, common_step_counter=0,
        episode_length_buf=torch.full((len(v_xy),), steps_in),
    )


def test_c6_rewards_make_slow_commands_count():
    from microduck_pretrain import followups as f

    # 0: asked 0.08 m/s, standing        1: asked 0.08, doing 0.08
    # 2: asked 0.40, doing 0.30           3: asked to stand, standing
    # 4: asked to turn 0.3 rad/s, not     5: asked to turn 0.3, turning 0.3
    env = _c6_env(
        v_xy=[(0, 0), (0.08, 0), (0.30, 0), (0, 0), (0, 0), (0, 0)],
        wz=[0, 0, 0, 0, 0, 0.3],
        cmd=[(0.08, 0, 0), (0.08, 0, 0), (0.40, 0, 0), (0, 0, 0), (0, 0, 0.3), (0, 0, 0.3)],
    )
    stall = f.careful_stall_cost(env).tolist()
    assert stall == pytest.approx([1.0, 0.0, 0.25, 0.0, 1.0, 0.0])
    lin = f.track_lin_relative(env)
    # standing on a slow command scores far worse than doing it, and no better
    # than being 25% slow on a fast one
    assert lin[0] < 0.1 and lin[1] == pytest.approx(1.0) and lin[2] > lin[0]
    assert lin[3] == 0.0  # stand command: the relative term is silent
    yaw = f.track_yaw_relative(env)
    assert yaw[4] < 0.1 and yaw[5] == pytest.approx(1.0)


def test_c6_velocity_is_smoothed_once_per_step():
    from microduck_pretrain import followups as f

    env = _c6_env(v_xy=[(0.0, 0.0)], wz=[0.0], cmd=[(0.1, 0, 0)])
    f.smoothed_base_velocity(env)  # initializes at 0
    env.scene["robot"].data.root_link_lin_vel_b[0, 0] = 0.2
    env.common_step_counter = 1
    v1, _ = f.smoothed_base_velocity(env)
    v2, _ = f.smoothed_base_velocity(env)  # same step: no second update
    alpha = 0.02 / f.C6_VEL_EMA_S
    assert v1[0, 0].item() == pytest.approx(0.2 * alpha)
    assert v2[0, 0].item() == pytest.approx(v1[0, 0].item())


@pytest.mark.slow
def test_c6_is_c3x_plus_careful_terms():
    from mjlab.tasks.registry import list_tasks

    from microduck_pretrain import followups as f

    assert f.C6_TASK in set(list_tasks())
    c3x, c6 = f.make_c3x_env_cfg(), f.make_c6_env_cfg()
    assert set(c6.rewards) - set(c3x.rewards) == {"careful_stall", "track_lin_relative", "track_yaw_relative"}
    assert c6.commands["twist"].ranges == c3x.commands["twist"].ranges
    assert "careful_stall" not in f.make_c3x_env_cfg().rewards  # C3x untouched
