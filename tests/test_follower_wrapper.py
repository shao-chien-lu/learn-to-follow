"""The C++ batched FollowerWrapper must reproduce the original python per-agent implementation exactly.

Both wrappers are run on identically seeded pogema envs with identical random actions; every observation
(obstacles/agents/xy/target_xy), reward and done flag is compared at every step, across episode resets.

Run: python -m pytest tests/test_follower_wrapper.py -v
"""
import numpy as np
import pytest

import planner_common  # noqa: F401  (puts the repo root and tests/reference on sys.path)
from env.create_env import create_env_base
from follower.preprocessing import FollowerWrapper, PreprocessorConfig
from follower.training_config import EnvironmentMazes
from legacy_wrapper import LegacyFollowerWrapper


def make_env(wrapper_cls, pre_cfg, map_name, num_agents, seed, max_steps):
    env_cfg = EnvironmentMazes()
    env_cfg.grid_config.num_agents = num_agents
    env_cfg.grid_config.map_name = map_name
    env_cfg.grid_config.max_episode_steps = max_steps
    env_cfg.grid_config.seed = seed
    return wrapper_cls(create_env_base(env_cfg), pre_cfg)


def assert_obs_equal(a, b):
    assert len(a) == len(b)
    for k, (oa, ob) in enumerate(zip(a, b)):
        assert oa.keys() == ob.keys()
        for key in oa:
            va, vb = oa[key], ob[key]
            if isinstance(va, np.ndarray):
                assert va.dtype == vb.dtype, (k, key)
                np.testing.assert_array_equal(va, vb, err_msg=f'agent {k} key {key}')
            else:
                assert tuple(va) == tuple(vb), (k, key)


@pytest.mark.parametrize('flags', [(True, True, True), (True, False, True), (False, True, False), (False, False, True)],
                         ids=lambda f: 'S%dD%dR%d' % tuple(map(int, f)))
@pytest.mark.parametrize('map_name', [r'mazes-.+', r'pico_.+', r'wfi_warehouse'])
def test_wrapper_matches_legacy(map_name, flags):
    cfg = PreprocessorConfig(use_static_cost=flags[0], use_dynamic_cost=flags[1], reset_dynamic_cost=flags[2],
                             network_input_radius=3, intrinsic_target_reward=0.01)
    legacy = make_env(LegacyFollowerWrapper, cfg, map_name, 24, seed=3, max_steps=70)
    new = make_env(FollowerWrapper, cfg, map_name, 24, seed=3, max_steps=70)
    rng = np.random.default_rng(0)

    rewards_seen = set()
    episode = 0
    obs_l, _ = legacy.reset(seed=episode)
    obs_n, _ = new.reset(seed=episode)
    assert_obs_equal(obs_l, obs_n)
    for step in range(220):
        actions = rng.integers(0, 5, size=24).tolist()
        if step % 3:  # often follow the plan, so agents actually reach sub-goals / targets
            for k, o in enumerate(obs_l):
                if o['xy'] != o['target_xy']:
                    nxt = legacy.prev_goals[k]
                    d = (nxt[0] - o['xy'][0], nxt[1] - o['xy'][1])
                    actions[k] = {(0, 0): 0, (-1, 0): 1, (1, 0): 2, (0, -1): 3, (0, 1): 4}.get(d, actions[k])
        obs_l, r_l, d_l, t_l, _ = legacy.step(list(actions))
        obs_n, r_n, d_n, t_n, _ = new.step(list(actions))
        assert_obs_equal(obs_l, obs_n)
        assert r_l == r_n
        assert d_l == d_n and t_l == t_n
        rewards_seen.update(r_l)
        if all(t_l) or all(d_l):
            episode += 1
            obs_l, _ = legacy.reset(seed=episode)
            obs_n, _ = new.reset(seed=episode)
            assert_obs_equal(obs_l, obs_n)
    assert 0.01 in rewards_seen  # the intrinsic reward path was actually exercised
    assert episode >= 2          # resets (new planners, cleared prev_goals) were exercised


def test_rewards_are_python_floats_and_obs_dtypes():
    cfg = PreprocessorConfig(network_input_radius=3)
    env = make_env(FollowerWrapper, cfg, r'mazes-.+', 8, seed=1, max_steps=30)
    obs, _ = env.reset(seed=0)
    obs, rew, *_ = env.step([0] * 8)
    assert all(type(r) is float for r in rew)
    assert all(o['obstacles'].dtype == np.float32 and o['agents'].dtype == np.float32 for o in obs)
    assert all(o['obstacles'].max() <= 1.0 and o['obstacles'].min() >= -1.0 for o in obs)
