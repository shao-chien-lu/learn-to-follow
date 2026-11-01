"""
Opt-in fix of pogema 1.3.0's soft collisions erasing agents from the positions grid (LTF_FIX_SOFT_POSITIONS=1).

Pogema.move_agents (soft) applies the resolved moves one agent at a time with Grid.move_without_checks, which frees
the agent's old cell and occupies its new one. When agent A enters the cell agent B leaves on the same step and A
moves first, B's move then frees the cell A has just entered: A vanishes from grid.positions, the source of every
agent's 'agents' observation channel, for that step (any next step, a stay included, rewrites A's cell, unless it is
wiped again, as happens on every step of a queue walking in move order). positions_xy, and so the moves and the throughput, stay
right; only what the agents see is wrong.

The fix rebuilds grid.positions from positions_xy after the moves, before the observations are made. It is a
process-wide patch of pogema's class, applied on import when the variable is set, so the training workers and the
evaluation's dask workers (which inherit the environment) all get it. Off by default, so that the earlier results
(on unpatched pogema) stay reproducible.
"""
import os

import numpy as np
from pogema.envs import Pogema

ENABLED = os.environ.get('LTF_FIX_SOFT_POSITIONS') == '1'


def rebuild_positions(grid):
    positions = np.zeros(grid.positions.shape, dtype=grid.positions.dtype)
    for agent_id, (x, y) in enumerate(grid.positions_xy):
        if grid.is_active[agent_id]:
            positions[x, y] = grid.config.OBSTACLE
    grid.positions = positions


def _move_agents_fixed(self, actions):
    _move_agents(self, actions)
    if self.grid.config.collision_system == 'soft':
        rebuild_positions(self.grid)


_move_agents = Pogema.move_agents
if ENABLED and Pogema.move_agents is not _move_agents_fixed:
    Pogema.move_agents = _move_agents_fixed
