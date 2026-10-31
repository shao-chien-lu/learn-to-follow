"""NeighbourMemoryWrapper: its channels on hand-built moves (the C++ side is checked against it in
test_cpp_inference.py), and their place in the network input.

Run: python -m pytest tests/test_neighbour_memory.py -v
"""
import numpy as np
import pytest
from pogema import pogema_v0

import planner_common  # noqa: F401  (puts the repo root and tests/reference on sys.path)
from env.create_env import ProvideGlobalObstacles
from follower.preprocessing import FollowerWrapper, NeighbourMemoryWrapper, PreprocessorConfig, wrap_preprocessors
from follower.training_config import DecMAPFConfig

STAY, UP, DOWN, LEFT, RIGHT = range(5)
R = 5  # obs_radius: the channels are (11, 11) before the crop, the agent's cell at (R, R)

# agent 0 moves right, then stays; agent 1 walks right past it; agent 2 stands still above-left of it
GRID = '\n'.join(['.' * 8] * 5)
AGENTS, TARGETS = [(3, 1), (3, 3), (1, 1)], [(0, 7), (0, 0), (4, 7)]
ACTIONS = [[RIGHT, RIGHT, STAY], [STAY, RIGHT, STAY]]


def make_env(wrap, **preprocessing):
    gc = DecMAPFConfig(map=GRID, agents_xy=AGENTS, targets_xy=TARGETS, num_agents=len(AGENTS), on_target='nothing',
                       max_episode_steps=32, obs_radius=R, seed=0)
    env = ProvideGlobalObstacles(pogema_v0(grid_config=gc))
    if wrap:
        return wrap_preprocessors(env, PreprocessorConfig(**preprocessing))
    return NeighbourMemoryWrapper(FollowerWrapper(env, PreprocessorConfig(network_input_radius=3)))


def run(env):
    obs, _ = env.reset()
    for actions in ACTIONS:
        obs, *_ = env.step(actions)
    return obs


def cells(channel):
    """{(di, dj) relative to the agent: value} of the nonzero cells."""
    return {(int(i) - R, int(j) - R): float(channel[i, j]) for i, j in zip(*np.nonzero(channel))}


def test_trail_and_wait():
    obs = run(make_env(wrap=False))[0]  # agent 0, now at (3, 2), was at (3, 2) and (3, 1)
    # agent 1 was at (3, 3) two steps ago and at (3, 4) one step ago; agent 2 is always at (1, 1); the agent's own
    # earlier cell (3, 1) is not another agent
    assert cells(obs['nb_trail']) == {(0, 1): 7 / 8, (0, 2): 1.0, (-2, -1): 1.0}
    # agent 2 has been there for both previous steps, agent 1 has just arrived at (3, 5)
    assert cells(obs['nb_wait']) == {(-2, -1): 2 / 8}


def test_nothing_before_the_first_step():
    obs, _ = make_env(wrap=False).reset()
    assert not obs[0]['nb_trail'].any() and not obs[0]['nb_wait'].any()


@pytest.mark.parametrize('mode, num_channels', [('none', 0), ('trail', 2)])
def test_channels_go_after_the_own_memory(mode, num_channels):
    obs = run(make_env(wrap=True, network_input_radius=R, use_memory=True, neighbour_memory=mode))[0]['obs']
    assert obs.shape == (4 + num_channels, 2 * R + 1, 2 * R + 1)
    reference = run(make_env(wrap=False))[0] if num_channels else {}
    for k, key in enumerate(['nb_trail', 'nb_wait'][:num_channels]):
        np.testing.assert_array_equal(obs[4 + k], reference[key])
    own = run(make_env(wrap=True, network_input_radius=R, use_memory=True))[0]['obs']
    np.testing.assert_array_equal(obs[:4], own)  # the existing channels are unchanged
