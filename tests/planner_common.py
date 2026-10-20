"""Shared helpers for tests/test_planner.py and tests/bench_planner.py.

`new_planner` is the optimised follower_cpp.planner; `ref_planner` is a frozen copy of the original
implementation (tests/reference/planner_ref.*), built on first import by cppimport.
"""
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'tests', 'reference'))

# noinspection PyUnresolvedReferences
import cppimport.import_hook  # noqa: E402
from follower_cpp.planner import planner as new_planner  # noqa: E402
from planner_ref import planner as ref_planner  # noqa: E402

RADIUS = 3  # FollowerLite obs radius (7x7); the grid is padded with `RADIUS` layers of obstacles, as pogema does


def _pad(inner, pad=RADIUS):
    return np.pad(inner, pad, constant_values=1)


def random_grid(size, density, seed):
    rng = np.random.default_rng(seed)
    return _pad((rng.random((size, size)) < density).astype(int))


def maze_grid(size, seed):
    """Perfect maze (DFS on odd cells) of odd `size`, so every free cell is connected."""
    rng = np.random.default_rng(seed)
    g = np.ones((size, size), dtype=int)
    stack = [(1, 1)]
    g[1, 1] = 0
    while stack:
        i, j = stack[-1]
        nbrs = [(i + di, j + dj, di, dj) for di, dj in ((2, 0), (-2, 0), (0, 2), (0, -2))
                if 0 < i + di < size - 1 and 0 < j + dj < size - 1 and g[i + di, j + dj]]
        if not nbrs:
            stack.pop()
            continue
        ni, nj, di, dj = nbrs[rng.integers(len(nbrs))]
        g[i + di // 2, j + dj // 2] = 0
        g[ni, nj] = 0
        stack.append((ni, nj))
    return _pad(g)


def warehouse_like_grid(h, w, seed):
    """Open floor with regular shelf blocks and a few random obstacles (large open areas, many ties)."""
    rng = np.random.default_rng(seed)
    g = np.zeros((h, w), dtype=int)
    g[2:h - 2:4, 3:w - 3] = 1
    g[3:h - 2:4, 3:w - 3] = 1
    g[:, ::9] = 0
    g[rng.random((h, w)) < 0.02] = 1
    return _pad(g)


def free_cells(grid):
    g = np.asarray(grid)
    r = RADIUS
    inner = np.argwhere(g[r:-r, r:-r] == 0) + r
    return [tuple(map(int, c)) for c in inner]


def occupation_window(rng, grid, offset):
    """Random 0/1 float64 (2r+1)x(2r+1) window around a random free cell; returns (array, cur_pos relative to offset)."""
    cells = free_cells(grid)
    pi, pj = cells[rng.integers(len(cells))]
    win = (rng.random((2 * RADIUS + 1, 2 * RADIUS + 1)) < 0.25).astype(np.float64)
    return win, (pi - RADIUS - offset[0], pj - RADIUS - offset[1])
