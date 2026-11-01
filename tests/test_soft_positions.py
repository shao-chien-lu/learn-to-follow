"""pogema 1.3.0's soft collisions erase an agent that enters a cell vacated on the same step by a later-moving agent
from the positions grid (the 'agents' channel); follower/pogema_fix.py rebuilds it when LTF_FIX_SOFT_POSITIONS=1.

The patch is applied on import, so each case runs in a fresh interpreter with or without the variable.
The C++ side (follower_cpp/follower.cpp set_state) is checked by running tests/test_cpp_inference.py with the variable
set: it must still build exactly the python observations.

Run: python -m pytest tests/test_soft_positions.py -v
"""
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent

# a queue of three agents walking right in a corridor, the tail first in pogema's move order: each one enters the
# cell the next agent leaves, so all but the leader are erased from the positions grid
QUEUE = """
import sys
sys.path.insert(0, {root!r})
import follower.pogema_fix
from pogema import GridConfig, pogema_v0
grid = '\\n'.join(['#' * 11] * 4 + ['.' * 11] * 2 + ['#' * 11] * 5)
config = GridConfig(map=grid, size=11, agents_xy=[(5, 5), (4, 2), (4, 3), (4, 4)],
                    targets_xy=[(5, 10), (4, 10), (5, 9), (5, 8)], num_agents=4, obs_radius=5,
                    observation_type='POMAPF', collision_system='soft', on_target='restart', max_episode_steps=64,
                    seed=0)
env = pogema_v0(config)
env.reset(seed=0)
obs, *_ = env.step([0, 4, 4, 4])
grid_xy = env.unwrapped.grid.get_agents_xy(ignore_borders=True)
assert [tuple(c) for c in grid_xy[1:]] == [(4, 3), (4, 4), (4, 5)], grid_xy  # every move happened
print(int(obs[0]['agents'].sum()) - 1)  # neighbours the agent at (5, 5) sees
"""


def visible_neighbours(fix):
    env = {k: v for k, v in os.environ.items() if k != 'LTF_FIX_SOFT_POSITIONS'}
    if fix:
        env['LTF_FIX_SOFT_POSITIONS'] = '1'
    out = subprocess.run([sys.executable, '-c', QUEUE.format(root=str(ROOT))], env=env, capture_output=True,
                         text=True, check=True)
    return int(out.stdout.split()[-1])


def test_unpatched_pogema_erases_followers():
    assert visible_neighbours(fix=False) == 1  # the bug: two of the three walkers are missing


def test_fix_shows_every_agent():
    assert visible_neighbours(fix=True) == 3


@pytest.mark.skipif(os.environ.get('LTF_FIX_SOFT_POSITIONS') != '1', reason='run with LTF_FIX_SOFT_POSITIONS=1')
def test_patch_is_active_in_this_process():
    import follower.pogema_fix as fix
    from pogema.envs import Pogema
    assert fix.ENABLED and Pogema.move_agents is fix._move_agents_fixed
