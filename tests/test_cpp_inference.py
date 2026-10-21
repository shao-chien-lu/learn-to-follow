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


@pytest.mark.parametrize('map_name', [r'mazes-.+', r'wfi_warehouse', r'pico_.+'])
def test_training_obs_equal_cpp_inference_obs(map_name):
    Follower, Config = _cpp_follower()
    n, radius = 24, 3
    env_cfg = EnvironmentMazes()
    env_cfg.grid_config.num_agents = n
    env_cfg.grid_config.map_name = map_name
    env_cfg.grid_config.max_episode_steps = 50
    env_cfg.grid_config.seed = 5
    env = wrap_preprocessors(create_env_base(env_cfg), PreprocessorConfig(network_input_radius=radius))
    rng = np.random.default_rng(1)
    for episode in range(2):
        obs, _ = env.reset(seed=episode)
        cfg = Config()
        cfg.obs_radius = env.grid_config.obs_radius
        cfg.num_threads = 1
        cfg.path_to_weights = ''
        algo = Follower()
        algo.init(cfg, np.array(env.get_global_obstacles(), dtype=np.float64), env.get_global_agents_xy())
        for step in range(50):
            algo.set_state([tuple(o['xy']) for o in obs], [tuple(o['target_xy']) for o in obs])
            for k, o in enumerate(obs):
                cpp = np.array(algo.observe(k, radius), dtype=np.float32)
                np.testing.assert_array_equal(cpp, o['obs'].ravel(), err_msg=f'episode {episode} step {step} agent {k}')
            obs, _, done, trunc, _ = env.step(rng.integers(0, 5, size=n).tolist())
            if all(done) or all(trunc):
                break
