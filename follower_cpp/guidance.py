"""Per-map guidance tables for the FollowerLite A* planner (planner::set_guidance).

A table is a float32 array (H, W, 4) over the grid the C++ Follower plans on (pogema's global obstacles, obstacle
border included, absolute coordinates): table[i, j, k] is the cost factor of the move from (i, j) in direction
DIRECTIONS[k] (planner.h's d4 order: east, south, north, west). It replaces the crisscross factor where the planner
applies it (with the model's crisscross_all semantics). crisscross_table(shape, ratio) is the table equivalent of
--crisscross_ratio ratio; all ones is equivalent to crisscross off.

Tables are stored as <dir>/<key>.npy with key = guidance_key(global obstacles, number of agents), so an evaluation can
pick the table of each map instance and agent count without knowing the map's name.
"""
import hashlib
from pathlib import Path

import numpy as np

DIRECTIONS = ((0, 1), (1, 0), (-1, 0), (0, -1))  # (di, dj) of planner.h's d4: east, south, north, west


def crisscross_table(shape, ratio):
    """planner.h's crisscross highways as a table: moves against them (east on odd rows, west on even rows, south on odd
    columns, north on even columns) get `ratio`, the others 1."""
    h, w = shape
    i = np.arange(h)[:, None] * np.ones((1, w), dtype=int)
    j = np.ones((h, 1), dtype=int) * np.arange(w)[None, :]
    against = np.stack([i % 2 == 1, j % 2 == 1, j % 2 == 0, i % 2 == 0], axis=-1)
    return np.where(against, np.float32(ratio), np.float32(1)).astype(np.float32)


def guidance_key(global_obstacles, num_agents):
    grid = np.asarray(global_obstacles, dtype=np.int8)
    digest = hashlib.sha1(np.array(grid.shape, dtype=np.int64).tobytes() + grid.tobytes()).hexdigest()[:16]
    return f'{digest}_n{int(num_agents)}'


def table_path(guidance_dir, global_obstacles, num_agents):
    return Path(guidance_dir) / f'{guidance_key(global_obstacles, num_agents)}.npy'


def save_table(guidance_dir, global_obstacles, num_agents, table):
    grid = np.asarray(global_obstacles)
    table = np.asarray(table, dtype=np.float32)
    assert table.shape == grid.shape + (4,), (table.shape, grid.shape)
    assert np.all(np.isfinite(table)) and np.all(table > 0)
    path = table_path(guidance_dir, grid, num_agents)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp.npy')
    np.save(tmp, table)
    tmp.replace(path)
    return path


def load_table(guidance_dir, global_obstacles, num_agents):
    """The table of this map and agent count (missing: an error, so that no evaluation silently runs without it)."""
    grid = np.asarray(global_obstacles)
    path = table_path(guidance_dir, grid, num_agents)
    if not path.exists():
        raise FileNotFoundError(f'no guidance table {path} for this map ({grid.shape}) and {num_agents} agents')
    table = np.load(path)
    assert table.shape == grid.shape + (4,) and table.dtype == np.float32, (table.shape, table.dtype)
    return table
