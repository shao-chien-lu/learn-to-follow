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


CC_OFF = {'crisscross_ratio': 1.0, 'crisscross_all': False}


@pytest.mark.parametrize('use_memory, neighbour_memory, radius, cc', [
    (False, 'none', 3, CC_OFF), (True, 'none', 3, CC_OFF), (True, 'trail', 3, CC_OFF), (False, 'trail', 5, CC_OFF),
    (True, 'none', 3, {'crisscross_ratio': 3.0, 'crisscross_all': False}),
    (True, 'trail', 3, {'crisscross_ratio': 1.5, 'crisscross_all': True})], ids=str)
@pytest.mark.parametrize('map_name', [r'mazes-.+', r'wfi_warehouse', r'pico_.+'])
def test_training_obs_equal_cpp_inference_obs(map_name, use_memory, neighbour_memory, radius, cc):
    Follower, Config = _cpp_follower()
    n = 24
    env_cfg = EnvironmentMazes()
    env_cfg.grid_config.num_agents = n
    env_cfg.grid_config.map_name = map_name
    env_cfg.grid_config.max_episode_steps = 50
    env_cfg.grid_config.seed = 5
    env = wrap_preprocessors(create_env_base(env_cfg), PreprocessorConfig(
        network_input_radius=radius, use_memory=use_memory, neighbour_memory=neighbour_memory, **cc))
    rng = np.random.default_rng(1)
    num_blocked = 0
    num_waiting = 0  # cells with an agent that has stayed there (nb_wait > 0)
    for episode in range(2):
        obs, _ = env.reset(seed=episode)
        cfg = Config()
        cfg.obs_radius = env.grid_config.obs_radius
        cfg.num_threads = 1
        cfg.path_to_weights = ''
        cfg.use_memory = use_memory
        cfg.neighbour_memory = neighbour_memory
        for key, value in cc.items():
            setattr(cfg, key, value)
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
                if neighbour_memory == 'trail':
                    num_waiting += np.count_nonzero(o['obs'][-1])
            prev_xy = [tuple(o['xy']) for o in obs]
            actions = rng.integers(0, 5, size=n).tolist()
            algo.set_actions(actions)
            obs, _, done, trunc, _ = env.step(list(actions))
            if all(done) or all(trunc):
                break
    if use_memory:
        assert num_blocked > 0, 'no agent was ever blocked, the memory channels were not exercised'
    if neighbour_memory == 'trail':
        assert num_waiting > 0, 'no agent ever stayed in view, nb_wait was not exercised'


CHANNELS = ['obstacles', 'agents', 'blocked', 'trail', 'nb_trail', 'nb_wait']  # network order with memory + nb


@pytest.mark.parametrize('ablate', ['blocked', 'trail', 'nb_trail', 'nb_wait', 'blocked,nb_wait'])
def test_ablate_channels(ablate):
    """ablate_channels zeroes exactly the named channels, in training (AblateChannelsWrapper) and in the C++ inference
    alike, and leaves every other channel as without the ablation."""
    Follower, Config = _cpp_follower()
    n, radius = 24, 3
    envs = []
    for names in ['', ablate]:
        env_cfg = EnvironmentMazes()
        env_cfg.grid_config.num_agents = n
        env_cfg.grid_config.max_episode_steps = 50
        env_cfg.grid_config.seed = 5
        envs.append(wrap_preprocessors(create_env_base(env_cfg), PreprocessorConfig(
            network_input_radius=radius, use_memory=True, neighbour_memory='trail', ablate_channels=names)))
    zeroed = [CHANNELS.index(name) for name in ablate.split(',')]
    kept = [c for c in range(len(CHANNELS)) if c not in zeroed]
    rng = np.random.default_rng(1)
    full, obs = envs[0].reset(seed=0)[0], envs[1].reset(seed=0)[0]
    cfg = Config()
    cfg.obs_radius = envs[1].grid_config.obs_radius
    cfg.num_threads = 1
    cfg.path_to_weights = ''
    cfg.use_memory = True
    cfg.neighbour_memory = 'trail'
    cfg.ablate_channels = ablate
    algo = Follower()
    algo.init(cfg, np.array(envs[1].get_global_obstacles(), dtype=np.float64), envs[1].get_global_agents_xy())
    num_nonzero = 0  # entries the ablation removed
    for step in range(30):
        algo.set_state([tuple(o['xy']) for o in obs], [tuple(o['target_xy']) for o in obs])
        for k, (f, o) in enumerate(zip(full, obs)):
            assert o['obs'].shape == f['obs'].shape == (len(CHANNELS), 2 * radius + 1, 2 * radius + 1)
            assert not o['obs'][zeroed].any()
            np.testing.assert_array_equal(o['obs'][kept], f['obs'][kept])
            num_nonzero += np.count_nonzero(f['obs'][zeroed])
            cpp = np.array(algo.observe(k, radius), dtype=np.float32)
            np.testing.assert_array_equal(cpp, o['obs'].ravel(), err_msg=f'step {step} agent {k}')
        actions = rng.integers(0, 5, size=n).tolist()
        algo.set_actions(actions)
        full = envs[0].step(list(actions))[0]
        obs = envs[1].step(list(actions))[0]
    assert num_nonzero > 0, f'{ablate} was always zero anyway, the test does not exercise the ablation'


def test_ablate_unknown_channel():
    env_cfg = EnvironmentMazes()
    env_cfg.grid_config.num_agents = 4
    with pytest.raises(ValueError, match='nb_wait'):
        wrap_preprocessors(create_env_base(env_cfg), PreprocessorConfig(use_memory=True, ablate_channels='nb_wait'))
    Follower, Config = _cpp_follower()
    cfg = Config()
    cfg.path_to_weights = ''
    cfg.use_memory = True
    cfg.ablate_channels = 'nb_wait'
    with pytest.raises(ValueError, match='nb_wait'):
        Follower().init(cfg, np.zeros((16, 16)), [(5, 5)])


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
