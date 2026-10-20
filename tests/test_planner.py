"""Correctness tests for the optimised A* planner (follower_cpp/planner.h).

1. Differential tests: the new planner must return *bit-identical* results to the original implementation
   (frozen in tests/reference) on random grids, mazes and warehouse-like maps, over multi-step episodes that
   exercise goal changes, dynamic-cost resets and repeated re-planning on the same object.
2. Independent oracle: the returned path is valid and has the optimal cost according to a Python Dijkstra.
3. Edge cases: start == goal, unreachable goal, abs_offset, penalty NaNs for isolated cells.

Run: python -m pytest tests/test_planner.py -v
"""
import heapq
import itertools

import numpy as np
import pytest

from planner_common import (RADIUS, free_cells, maze_grid, new_planner, occupation_window, random_grid,
                            ref_planner, warehouse_like_grid)

FLAGS = list(itertools.product([True, False], repeat=3))  # use_static, use_dynamic, reset_dynamic

GRIDS = {
    'random20_d10': lambda s: random_grid(20, 0.10, s),
    'random20_d30': lambda s: random_grid(20, 0.30, s),   # often disconnected -> unreachable goals
    'random20_d45': lambda s: random_grid(20, 0.45, s),   # many isolated cells
    'maze31': lambda s: maze_grid(31, s),
    'maze65': lambda s: maze_grid(65, s),
    'warehouse': lambda s: warehouse_like_grid(33, 46, s),
}


def make_pair(grid, flags, offset, static_penalties=True):
    g = grid.tolist()
    new, ref = new_planner(g, *flags), ref_planner(g, *flags)
    for p in (new, ref):
        p.set_abs_start(offset)
    if flags[0] and static_penalties:
        pen_new = new_planner(g, *flags).precompute_penalty_matrix(RADIUS)
        pen_ref = ref_planner(g, *flags).precompute_penalty_matrix(RADIUS)
        np.testing.assert_array_equal(np.array(pen_new), np.array(pen_ref))
        new.set_penalties(pen_new)
        ref.set_penalties(pen_ref)
    return new, ref


def assert_same_outputs(new, ref):
    assert new.get_path() == ref.get_path()
    assert new.get_next_node() == ref.get_next_node()


@pytest.mark.parametrize('grid_name', GRIDS)
@pytest.mark.parametrize('seed', range(3))
def test_precompute_penalty_matrix_matches_reference(grid_name, seed):
    g = GRIDS[grid_name](seed).tolist()
    for radius in (1, RADIUS):
        a = np.array(new_planner(g, True, True, True).precompute_penalty_matrix(radius))
        b = np.array(ref_planner(g, True, True, True).precompute_penalty_matrix(radius))
        np.testing.assert_array_equal(a, b)  # also treats NaN == NaN


@pytest.mark.parametrize('flags', FLAGS, ids=lambda f: 'S%dD%dR%d' % tuple(map(int, f)))
@pytest.mark.parametrize('grid_name', GRIDS)
@pytest.mark.parametrize('seed', range(3))
def test_episode_matches_reference(grid_name, seed, flags):
    """Agent walks along the planned path, picks a new goal on arrival; observations are random each step."""
    rng = np.random.default_rng(1000 + seed)
    grid = GRIDS[grid_name](seed)
    cells = free_cells(grid)
    offset = (int(rng.integers(0, 4)), int(rng.integers(0, 4)))
    new, ref = make_pair(grid, flags, offset)

    pos = cells[rng.integers(len(cells))]
    goal = cells[rng.integers(len(cells))]
    for step in range(60):
        rel = lambda c: (c[0] - offset[0], c[1] - offset[1])
        if rng.random() < 0.7:  # not every step has an observation update
            win, cur = occupation_window(rng, grid, offset)
            new.update_occupations(win, cur, rel(goal))
            ref.update_occupations(win, cur, rel(goal))
        new.update_path(rel(pos), rel(goal))
        ref.update_path(rel(pos), rel(goal))
        assert_same_outputs(new, ref)

        path = ref.get_path()
        if len(path) > 1:
            nxt = path[1]
            pos = (nxt[0] + offset[0], nxt[1] + offset[1])
        if pos == goal or rng.random() < 0.1:
            goal = cells[rng.integers(len(cells))]
        if rng.random() < 0.05:  # teleport (e.g. fresh start after restart)
            pos = cells[rng.integers(len(cells))]


@pytest.mark.parametrize('flags', FLAGS[:4], ids=lambda f: 'S%dD%dR%d' % tuple(map(int, f)))
def test_many_random_queries_same_goal_and_new_goal(flags):
    """Replan repeatedly with the same goal (reuses h_values) and with different starts."""
    rng = np.random.default_rng(7)
    grid = maze_grid(31, 1)
    cells = free_cells(grid)
    new, ref = make_pair(grid, flags, (0, 0))
    goal = cells[0]
    for q in range(300):
        if q % 25 == 0:
            goal = cells[rng.integers(len(cells))]
        start = cells[rng.integers(len(cells))]
        new.update_path(start, goal)
        ref.update_path(start, goal)
        assert_same_outputs(new, ref)


def test_update_occupations_accepts_non_float64_and_noncontiguous_arrays():
    grid = random_grid(20, 0.1, 0)
    cells = free_cells(grid)
    start, goal = cells[0], cells[-1]
    rng = np.random.default_rng(0)
    base = (rng.random((2 * RADIUS + 1, 2 * RADIUS + 1)) < 0.4)
    variants = [base.astype(np.float64), base.astype(np.int64), base.astype(np.float32), base.astype(bool),
                np.asfortranarray(base.astype(np.float64)), base.astype(np.float64).T.copy().T]
    results = []
    for arr in variants:
        p = new_planner(grid.tolist(), True, True, True)
        p.update_occupations(arr, (3, 3), goal)
        p.update_path(start, goal)
        results.append(p.get_path())
    ref = ref_planner(grid.tolist(), True, True, True)
    ref.update_occupations(base.astype(np.float64), (3, 3), goal)
    ref.update_path(start, goal)
    assert all(r == ref.get_path() for r in results)


# ---------------------------------------------------------------- independent oracle (Python Dijkstra)

def dijkstra_cost(grid, cost, start, goal):
    H, W = grid.shape
    dist = np.full((H, W), np.inf)
    dist[start] = 0.0
    pq = [(0.0, start)]
    while pq:
        d, (i, j) = heapq.heappop(pq)
        if d > dist[i, j]:
            continue
        if (i, j) == goal:
            return d
        for di, dj in ((0, 1), (1, 0), (-1, 0), (0, -1)):
            n = (i + di, j + dj)
            if grid[n] == 0 and d + cost[n] < dist[n]:
                dist[n] = d + cost[n]
                heapq.heappush(pq, (dist[n], n))
    return np.inf


@pytest.mark.parametrize('flags', [(True, True, True), (True, False, True), (False, True, True), (False, False, True)],
                         ids=['static+dyn', 'static', 'dyn', 'uniform'])
@pytest.mark.parametrize('grid_name', ['random20_d30', 'maze31', 'warehouse'])
def test_path_is_valid_and_optimal(grid_name, flags):
    use_static, use_dynamic, _ = flags
    rng = np.random.default_rng(42)
    grid = GRIDS[grid_name](5)
    cells = free_cells(grid)
    penalties = np.ones(grid.shape)
    if use_static:
        penalties = np.array(new_planner(grid.tolist(), *flags).precompute_penalty_matrix(RADIUS))

    for _ in range(40):
        start, goal = cells[rng.integers(len(cells))], cells[rng.integers(len(cells))]
        win, cur = occupation_window(rng, grid, (0, 0))
        occ = np.zeros(grid.shape)
        ii, jj = np.nonzero(win)
        occ[ii + cur[0], jj + cur[1]] += 1

        p = new_planner(grid.tolist(), *flags)
        if use_static:
            p.set_penalties(penalties.tolist())
        p.update_occupations(win, cur, goal)
        p.update_path(start, goal)
        path = p.get_path()

        cost = (penalties if use_static else np.ones(grid.shape)) + (occ if use_dynamic else 0)
        best = dijkstra_cost(grid, cost, start, goal)
        if start == goal or not np.isfinite(best):
            assert path == []
            assert p.get_next_node() == ((10 ** 9, 10 ** 9), (10 ** 9, 10 ** 9))
            continue
        assert path[0] == start and path[-1] == goal
        for a, b in zip(path, path[1:]):
            assert abs(a[0] - b[0]) + abs(a[1] - b[1]) == 1 and grid[b] == 0
        got = sum(cost[c] for c in path[1:])
        if use_static:
            # With static costs the heuristic (reverse Dijkstra charging the *entered* cell) is slightly
            # inadmissible, so the original planner is only near-optimal too; never better than optimal.
            assert best * (1 - 1e-5) <= got <= best * 1.05
        else:
            assert got == pytest.approx(best, rel=1e-4)
        assert p.get_next_node() == (start, path[1])


# ---------------------------------------------------------------- edge cases

@pytest.mark.parametrize('make', [new_planner, ref_planner], ids=['new', 'ref'])
def test_start_equals_goal(make):
    grid = maze_grid(15, 0)
    c = free_cells(grid)[3]
    p = make(grid.tolist(), True, True, True)
    p.update_path(c, c)
    assert p.get_path() == []
    assert p.get_next_node() == ((10 ** 9, 10 ** 9), (10 ** 9, 10 ** 9))


def test_unreachable_goal_matches_reference():
    inner = np.zeros((9, 9), dtype=int)
    inner[:, 4] = 1  # wall splitting the map in two
    grid = np.pad(inner, RADIUS, constant_values=1)
    new, ref = make_pair(grid, (True, True, True), (0, 0))
    new.update_path((RADIUS, RADIUS), (RADIUS + 8, RADIUS + 8))
    ref.update_path((RADIUS, RADIUS), (RADIUS + 8, RADIUS + 8))
    assert new.get_path() == ref.get_path() == []
    assert new.get_next_node() == ref.get_next_node()


def test_adjacent_start_goal_and_path_changes_after_occupation():
    grid = np.pad(np.zeros((5, 5), dtype=int), RADIUS, constant_values=1)
    new, ref = make_pair(grid, (False, True, True), (0, 0))
    s, mid, g = (RADIUS, RADIUS), (RADIUS, RADIUS + 1), (RADIUS, RADIUS + 2)
    for p in (new, ref):
        p.update_path(s, g)
    assert new.get_path() == ref.get_path() == [s, mid, g]

    # block the middle cell with many "agents" (goal unchanged -> no reset): path must detour, identically in both
    win = np.zeros((2 * RADIUS + 1,) * 2)
    win[mid] = 1
    for p in (new, ref):
        for _ in range(5):
            p.update_occupations(win, (0, 0), g)
        p.update_path(s, g)
    assert new.get_path() == ref.get_path()
    assert len(new.get_path()) == 5 and mid not in new.get_path()


def test_isolated_cell_penalty_is_nan_in_both():
    inner = np.ones((5, 5), dtype=int)
    inner[2, 2] = 0
    inner[0, 0] = 0
    inner[0, 1] = 0
    grid = np.pad(inner, RADIUS, constant_values=1).tolist()
    a = np.array(new_planner(grid, True, True, True).precompute_penalty_matrix(RADIUS))
    b = np.array(ref_planner(grid, True, True, True).precompute_penalty_matrix(RADIUS))
    np.testing.assert_array_equal(a, b)
    assert np.isnan(a[RADIUS + 2, RADIUS + 2])
