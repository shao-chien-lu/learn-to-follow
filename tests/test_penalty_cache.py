"""Tests for the penalty cache in follower/planning.py.

Run: python -m pytest tests/test_penalty_cache.py -v
"""
import numpy as np
import pytest

from planner_common import RADIUS, free_cells, maze_grid, random_grid, warehouse_like_grid
import follower.planning as planning
from follower.planning import Planner, PlannerConfig, get_penalties

MAPS = {
    'maze31': lambda: maze_grid(31, 3),
    'random20': lambda: random_grid(20, 0.2, 1),
    'warehouse': lambda: warehouse_like_grid(33, 46, 2),
}


@pytest.fixture(autouse=True)
def clean_cache():
    planning._PENALTY_CACHE.clear()
    yield
    planning._PENALTY_CACHE.clear()


def run_episode(grid, seed, steps=40, num_agents=6):
    """Drive a Planner like the env wrapper does; returns the list of get_path() results per step."""
    rng = np.random.default_rng(seed)
    cells = free_cells(grid)
    pick = lambda: cells[rng.integers(len(cells))]
    offsets = [(int(rng.integers(0, 4)), int(rng.integers(0, 4))) for _ in range(num_agents)]
    pos = [pick() for _ in range(num_agents)]
    goals = [pick() for _ in range(num_agents)]
    planner = Planner(PlannerConfig())
    planner.add_grid_obstacles(grid.tolist(), offsets)
    out = []
    for _ in range(steps):
        obs = []
        for k in range(num_agents):
            agents = (rng.random((2 * RADIUS + 1,) * 2) < 0.2).astype(np.float64)
            agents[RADIUS, RADIUS] = 1
            obs.append({'obstacles': np.zeros((2 * RADIUS + 1,) * 2), 'agents': agents,
                        'xy': (pos[k][0] - offsets[k][0], pos[k][1] - offsets[k][1]),
                        'target_xy': (goals[k][0] - offsets[k][0], goals[k][1] - offsets[k][1])})
        planner.update(obs)
        paths = planner.get_path()
        out.append(paths)
        for k, path in enumerate(paths):
            if len(path) > 1:
                pos[k] = (path[1][0] + offsets[k][0], path[1][1] + offsets[k][1])
            if pos[k] == goals[k]:
                goals[k] = pick()
    return out


@pytest.mark.parametrize('name', MAPS)
def test_cached_episode_identical_to_uncached(name, monkeypatch):
    grid = MAPS[name]()
    cached_cold = run_episode(grid, seed=0)        # fills the cache
    assert len(planning._PENALTY_CACHE) == 1
    cached_warm = run_episode(grid, seed=0)        # served from the cache
    assert len(planning._PENALTY_CACHE) == 1

    # uncached: wipe the cache before every lookup, so every Planner recomputes penalties
    orig = planning.get_penalties
    monkeypatch.setattr(planning, 'get_penalties', lambda *a, **k: (planning._PENALTY_CACHE.clear(), orig(*a, **k))[1])
    uncached = run_episode(grid, seed=0)

    assert cached_cold == uncached
    assert cached_warm == uncached
    assert any(len(p) > 1 for step in uncached for p in step)  # the episode actually planned something


def test_hit_returns_same_object_and_skips_recompute(monkeypatch):
    grid = MAPS['maze31']().tolist()
    cfg = PlannerConfig()
    first = get_penalties(grid, RADIUS, cfg)

    def boom(*a, **k):
        raise AssertionError('penalties recomputed on a cache hit')
    monkeypatch.setattr(planning, 'planner', boom)
    assert get_penalties([row[:] for row in grid], RADIUS, cfg) is first   # equal-valued copy still hits


def test_key_distinguishes_map_radius_and_shape():
    cfg = PlannerConfig()
    g = MAPS['maze31']().tolist()
    a = get_penalties(g, RADIUS, cfg)
    assert get_penalties(g, RADIUS - 1, cfg) is not a
    g2 = [row[:] for row in g]
    g2[10][10] ^= 1
    assert get_penalties(g2, RADIUS, cfg) is not a
    x = np.pad(np.zeros((4, 6), dtype=int), RADIUS, constant_values=1)
    y = x.reshape(-1).reshape(x.shape[1], x.shape[0])   # same bytes, transposed shape
    assert get_penalties(x.tolist(), 1, cfg) is not get_penalties(y.tolist(), 1, cfg)
    assert len(planning._PENALTY_CACHE) == 5


def test_eviction_is_bounded_and_fifo(monkeypatch):
    monkeypatch.setattr(planning, '_PENALTY_CACHE_MAX', 3)
    cfg = PlannerConfig()
    grids = [random_grid(12, 0.2, s).tolist() for s in range(5)]
    for g in grids:
        get_penalties(g, RADIUS, cfg)
        assert len(planning._PENALTY_CACHE) <= 3
    keys = list(planning._PENALTY_CACHE)
    assert keys == [(np.asarray(g, dtype=np.int8).tobytes(), len(g), RADIUS) for g in grids[2:]]
    # evicted entry is recomputed to the same values
    again = get_penalties(grids[0], RADIUS, cfg)
    np.testing.assert_array_equal(np.array(again), np.array(planning.planner(grids[0], True, True, True)
                                                           .precompute_penalty_matrix(RADIUS)))
