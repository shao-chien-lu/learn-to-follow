"""Per-map guidance tables (planner::set_guidance, follower_cpp/guidance.py).

1. The crisscross pattern as a table is bit-identical to set_crisscross(ratio, all); an all-ones table is bit-identical
   to crisscross off and to the original planner (tests/reference).
2. Oracle: with any table (factors below 1 included) the path is optimal for the directed move costs of planner.h.
3. The C++ Follower builds the same observations with guidance_table == crisscross table as with the crisscross flags.

Run: python -m pytest tests/test_guidance.py -v
"""
import itertools
import os

import numpy as np
import pytest

from planner_common import free_cells, new_planner, occupation_window
from test_planner import FLAGS, GRIDS, directed_dijkstra_cost, make_pair, assert_same_outputs
from follower_cpp.guidance import DIRECTIONS, crisscross_table, guidance_key, load_table, save_table


def _episode(new, ref, grid, offset, seed, steps=60):
    """Drive both planners through the same multi-step episode (goal changes, occupations) and compare every path."""
    rng = np.random.default_rng(seed)
    cells = free_cells(grid)
    rel = lambda c: (c[0] - offset[0], c[1] - offset[1])
    goal = cells[0]
    for step in range(steps):
        if step % 15 == 0:
            goal = cells[rng.integers(len(cells))]
        win, cur = occupation_window(rng, grid, offset)
        new.update_occupations(win, cur, rel(goal))
        ref.update_occupations(win, cur, rel(goal))
        start = cells[rng.integers(len(cells))]
        new.update_path(rel(start), rel(goal))
        ref.update_path(rel(start), rel(goal))
        assert_same_outputs(new, ref)


@pytest.mark.parametrize('flags', FLAGS, ids=lambda f: 'S%dD%dR%d' % tuple(map(int, f)))
@pytest.mark.parametrize('cc_all', [False, True])
@pytest.mark.parametrize('ratio', [2.0, 1.5])
@pytest.mark.parametrize('grid_name', ['maze31', 'warehouse', 'random20_d30'])
def test_crisscross_table_equals_crisscross(grid_name, ratio, cc_all, flags):
    grid = GRIDS[grid_name](1)
    offset = (2, 1)
    with_table, with_cc = make_pair(grid, flags, offset)[0], make_pair(grid, flags, offset)[0]
    with_cc.set_crisscross(ratio, cc_all)
    with_table.set_crisscross(1.0, cc_all)  # only the 'all' semantics count with a table
    with_table.set_guidance(crisscross_table(grid.shape, ratio).ravel().tolist())
    _episode(with_table, with_cc, grid, offset, seed=7)


@pytest.mark.parametrize('flags', FLAGS, ids=lambda f: 'S%dD%dR%d' % tuple(map(int, f)))
@pytest.mark.parametrize('cc_all', [False, True])
def test_ones_table_equals_reference(flags, cc_all):
    grid = GRIDS['warehouse'](1)
    new, ref = make_pair(grid, flags, (1, 2))
    new.set_crisscross(3.0, cc_all)  # replaced by the table
    new.set_guidance(np.ones(grid.shape + (4,), np.float32).ravel().tolist())
    _episode(new, ref, grid, (1, 2), seed=5)


def test_empty_table_turns_guidance_off():
    grid = GRIDS['maze31'](0)
    flags = (True, True, True)
    new, ref = make_pair(grid, flags, (0, 0))
    new.set_crisscross(2.0, True)
    ref_cc = make_pair(grid, flags, (0, 0))[0]
    ref_cc.set_crisscross(2.0, True)
    new.set_guidance(np.full(grid.shape + (4,), 3.0, np.float32).ravel().tolist())
    new.set_guidance([])
    _episode(new, ref_cc, grid, (0, 0), seed=3)


@pytest.mark.parametrize('lo, hi', [(1.0, 4.0), (0.5, 8.0), (0.25, 2.0)])
@pytest.mark.parametrize('use_dynamic, cc_all', [(False, False), (True, False), (True, True)])
@pytest.mark.parametrize('grid_name', ['random20_d10', 'maze31', 'warehouse'])
def test_guidance_against_oracle(grid_name, use_dynamic, cc_all, lo, hi):
    """Random tables, factors below 1 included (h is scaled down then); without static costs the heuristic is
    admissible, so paths must be optimal for the directed costs (1 + dyn) * f (all) or f + dyn, f = table[u][k]."""
    rng = np.random.default_rng(17)
    grid = GRIDS[grid_name](2)
    cells = free_cells(grid)
    offset = (3, 1)
    rel = lambda c: (c[0] - offset[0], c[1] - offset[1])
    num_checked = 0
    for _ in range(30):
        table = np.exp(rng.uniform(np.log(lo), np.log(hi), size=grid.shape + (4,))).astype(np.float32)
        start, goal = cells[rng.integers(len(cells))], cells[rng.integers(len(cells))]
        win, cur = occupation_window(rng, grid, offset)
        occ = np.zeros(grid.shape)
        ii, jj = np.nonzero(win)
        occ[ii + cur[0] + offset[0], jj + cur[1] + offset[1]] += 1

        p = new_planner(grid.tolist(), False, use_dynamic, True)
        p.set_abs_start(offset)
        p.set_crisscross(1.0, cc_all)
        p.set_guidance(table.ravel().tolist())
        p.update_occupations(win, cur, rel(goal))
        p.update_path(rel(start), rel(goal))
        path = [(a + offset[0], b + offset[1]) for a, b in p.get_path()]

        def move_cost(u, v):
            factor = float(table[u[0], u[1], DIRECTIONS.index((v[0] - u[0], v[1] - u[1]))])
            dyn = occ[v] if use_dynamic else 0.0
            return (1.0 + dyn) * factor if cc_all else factor + dyn

        best = directed_dijkstra_cost(grid, move_cost, start, goal)
        if start == goal or not np.isfinite(best):
            assert path == []
            continue
        assert path[0] == start and path[-1] == goal
        for a, b in zip(path, path[1:]):
            assert abs(a[0] - b[0]) + abs(a[1] - b[1]) == 1 and grid[b] == 0
        assert sum(move_cost(a, b) for a, b in zip(path, path[1:])) == pytest.approx(best, rel=1e-4)
        num_checked += 1
    assert num_checked > 20


def test_guidance_rejects_bad_tables():
    grid = GRIDS['maze31'](0)
    p = new_planner(grid.tolist(), True, True, True)
    with pytest.raises(ValueError):
        p.set_guidance([1.0] * (grid.size * 4 - 1))
    for bad in (0.0, -1.0, float('nan'), float('inf')):
        t = np.ones(grid.size * 4, np.float32)
        t[5] = bad
        with pytest.raises(ValueError):
            p.set_guidance(t.tolist())


def test_crisscross_table_pattern():
    """The table helper follows planner.h's highways: east on even rows, west on odd rows, south on even columns,
    north on odd columns are free."""
    t = crisscross_table((6, 7), 2.0)
    for i, j in itertools.product(range(6), range(7)):
        assert t[i, j, 0] == (1.0 if i % 2 == 0 else 2.0)  # east
        assert t[i, j, 3] == (1.0 if i % 2 == 1 else 2.0)  # west
        assert t[i, j, 1] == (1.0 if j % 2 == 0 else 2.0)  # south
        assert t[i, j, 2] == (1.0 if j % 2 == 1 else 2.0)  # north


def test_save_load_by_key(tmp_path):
    grid = GRIDS['maze31'](0)
    t = crisscross_table(grid.shape, 2.0)
    save_table(tmp_path, grid, 64, t)
    np.testing.assert_array_equal(load_table(tmp_path, grid.tolist(), 64), t)
    assert guidance_key(grid, 64) != guidance_key(grid, 32)
    other = grid.copy()
    other[5, 5] = 1 - other[5, 5]
    assert guidance_key(other, 64) != guidance_key(grid, 64)
    with pytest.raises(FileNotFoundError):
        load_table(tmp_path, grid, 32)


@pytest.mark.parametrize('map_name', [r'mazes-.+', r'wfi_warehouse'])
def test_cpp_follower_crisscross_table_equals_crisscross(map_name):
    """The C++ Follower with guidance_table == the crisscross table (ratio 2, all) draws the same A* paths into the
    observations as with crisscross_ratio 2 / crisscross_all, over episodes with random actions."""
    from test_cpp_inference import _cpp_follower
    from env.create_env import create_env_base
    from follower.training_config import EnvironmentMazes
    from follower.preprocessing import PreprocessorConfig, wrap_preprocessors
    Follower, Config = _cpp_follower()
    n, radius = 32, 5
    env_cfg = EnvironmentMazes()
    env_cfg.grid_config.num_agents = n
    env_cfg.grid_config.map_name = map_name
    env_cfg.grid_config.max_episode_steps = 40
    env_cfg.grid_config.seed = 3
    env = wrap_preprocessors(create_env_base(env_cfg), PreprocessorConfig(
        network_input_radius=radius, use_memory=True, neighbour_memory='trail', crisscross_ratio=2.0,
        crisscross_all=True))
    rng = np.random.default_rng(2)
    num_path_cells = 0
    for episode in range(2):
        obs, _ = env.reset(seed=episode)
        grid = np.array(env.get_global_obstacles(), dtype=np.float64)
        algos = []
        for use_table in (False, True):
            cfg = Config()
            cfg.obs_radius = env.grid_config.obs_radius
            cfg.num_threads = 1
            cfg.path_to_weights = ''
            cfg.use_memory = True
            cfg.neighbour_memory = 'trail'
            cfg.crisscross_all = True
            if use_table:
                cfg.guidance_table = crisscross_table(grid.shape, 2.0).ravel().tolist()
            else:
                cfg.crisscross_ratio = 2.0
            algos.append(Follower())
            algos[-1].init(cfg, grid, env.get_global_agents_xy())
        for step in range(40):
            for a in algos:
                a.set_state([tuple(o['xy']) for o in obs], [tuple(o['target_xy']) for o in obs])
            for k, o in enumerate(obs):
                cc, table = (np.array(a.observe(k, radius), dtype=np.float32) for a in algos)
                np.testing.assert_array_equal(table, cc)
                np.testing.assert_array_equal(table, o['obs'].ravel())  # and the training wrappers agree
                num_path_cells += int(np.count_nonzero(o['obs'][0] == 1))
            actions = rng.integers(0, 5, size=n).tolist()
            for a in algos:
                a.set_actions(actions)
            obs, _, done, trunc, _ = env.step(list(actions))
            if all(done) or all(trunc):
                break
    assert num_path_cells > 0
