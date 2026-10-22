"""The C++ inference (follower_cpp/follower.cpp, used by eval_lite.py) must build exactly the observations the network
was trained on (python wrappers + multi_planner), on identically seeded pogema envs with random actions.

Run: python -m pytest tests/test_cpp_inference.py -v
"""
import os

import numpy as np
import pytest

import planner_common  # noqa: F401  (puts the repo root and tests/reference on sys.path)
from env.create_env import create_env_base
from follower.preprocessing import PreprocessorConfig, wrap_preprocessors
from follower.training_config import EnvironmentMazes


def _cpp_follower():
    os.environ.setdefault('CPATH', os.path.expanduser('~/local/include'))
    os.environ.setdefault('LIBRARY_PATH', os.path.expanduser('~/local/lib'))
    try:
        from follower_cpp.follower import Follower
        from follower_cpp.config import Config
    except Exception as e:  # onnxruntime headers / libs missing
        pytest.skip(f'cannot build follower_cpp.follower: {e}')
    return Follower, Config


@pytest.mark.parametrize('use_memory', [False, True])
@pytest.mark.parametrize('map_name', [r'mazes-.+', r'wfi_warehouse', r'pico_.+'])
def test_training_obs_equal_cpp_inference_obs(map_name, use_memory):
    Follower, Config = _cpp_follower()
    n, radius = 24, 3
    env_cfg = EnvironmentMazes()
    env_cfg.grid_config.num_agents = n
    env_cfg.grid_config.map_name = map_name
    env_cfg.grid_config.max_episode_steps = 50
    env_cfg.grid_config.seed = 5
    env = wrap_preprocessors(create_env_base(env_cfg), PreprocessorConfig(network_input_radius=radius,
                                                                          use_memory=use_memory))
    rng = np.random.default_rng(1)
    num_blocked = 0
    for episode in range(2):
        obs, _ = env.reset(seed=episode)
        cfg = Config()
        cfg.obs_radius = env.grid_config.obs_radius
        cfg.num_threads = 1
        cfg.path_to_weights = ''
        cfg.use_memory = use_memory
        algo = Follower()
        algo.init(cfg, np.array(env.get_global_obstacles(), dtype=np.float64), env.get_global_agents_xy())
        prev_xy, actions = None, None
        for step in range(50):
            algo.set_state([tuple(o['xy']) for o in obs], [tuple(o['target_xy']) for o in obs])
            for k, o in enumerate(obs):
                cpp = np.array(algo.observe(k, radius), dtype=np.float32)
                np.testing.assert_array_equal(cpp, o['obs'].ravel(), err_msg=f'episode {episode} step {step} agent {k}')
                if use_memory and prev_xy is not None:
                    num_blocked += _check_memory_channels(o, prev_xy[k], actions[k], radius)
            prev_xy = [tuple(o['xy']) for o in obs]
            actions = rng.integers(0, 5, size=n).tolist()
            algo.set_actions(actions)
            obs, _, done, trunc, _ = env.step(list(actions))
            if all(done) or all(trunc):
                break
    if use_memory:
        assert num_blocked > 0, 'no agent was ever blocked, the memory channels were not exercised'


def _check_memory_channels(o, prev_xy, action, radius):
    """The memory channels against their definition (MemoryWrapper) for the step prev_xy --action--> o['xy'].
    Returns 1 if the agent was blocked."""
    blocked, trail = o['obs'][2], o['obs'][3]
    di, dj = int(prev_xy[0] - o['xy'][0]), int(prev_xy[1] - o['xy'][1])  # xy is float32 after the preprocessing
    assert trail[radius + di, radius + dj] == 1.0  # the previous position is the most recent trail entry
    if (di, dj) != (0, 0):
        assert not blocked.any()
        return 0
    move = [[0, 0], [-1, 0], [1, 0], [0, -1], [0, 1]][action]
    assert blocked[radius + move[0], radius + move[1]] > 0
    assert np.count_nonzero(blocked) == 1
    return 1
