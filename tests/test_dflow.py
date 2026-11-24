"""Direction-aware dynamic cost from the observed flow of other agents (planner::set_dflow, Follower::add_observed_flow).

1. Flags off is bit-identical: the planner with set_dflow(0) (and ignored add_observed_move calls) equals the original
   reference planner; the C++ Follower with dflow 'off' builds the same observations as the base guidance setting.
2. Flow accumulation: a move u -> v is credited to u and v in its d4 direction; the Follower infers the moves of the
   other agents from an agent's own last two views (hand-built scenes: a move, a stay, an ambiguous source, an unseen
   source) and, on real episodes, every inferred move is a true move of some agent.
3. Reset: the flow is cleared exactly when num_occupations is (a new goal with reset_dynamic_cost), and kept otherwise.
4. Oracle: in each mode the path is optimal for the directed costs defined in planner.h; all costs stay >= 0.

Run: LTF_FIX_SOFT_POSITIONS=1 python -m pytest tests/test_dflow.py -v
"""
import os

import numpy as np
import pytest

from planner_common import free_cells, new_planner, occupation_window
from test_planner import FLAGS, GRIDS, directed_dijkstra_cost, make_pair, assert_same_outputs
from follower_cpp.guidance import DIRECTIONS

MODES = {'off': 0, 'add': 1, 'split': 2, 'ggboost': 3}


# ---------------------------------------------------------------------------------------------------------------- 1.
@pytest.mark.parametrize('flags', FLAGS, ids=lambda f: 'S%dD%dR%d' % tuple(map(int, f)))
@pytest.mark.parametrize('grid_name', ['maze31', 'warehouse', 'random20_d30'])
def test_dflow_off_equals_reference(grid_name, flags):
    rng = np.random.default_rng(3)
    grid = GRIDS[grid_name](1)
    cells = free_cells(grid)
    offset = (2, 1)
    rel = lambda c: (c[0] - offset[0], c[1] - offset[1])
    new, ref = make_pair(grid, flags, offset)
    new.set_dflow(1, 1.0, 1.0)
    new.set_dflow(0, 0.0, 2.0)  # back to off: the flow is dropped and moves are ignored
    goal = cells[0]
    for step in range(60):
        if step % 15 == 0:
            goal = cells[rng.integers(len(cells))]
        win, cur = occupation_window(rng, grid, offset)
        new.update_occupations(win, cur, rel(goal))
        ref.update_occupations(win, cur, rel(goal))
        u = cells[rng.integers(len(cells))]
        new.add_observed_move(rel(u), rel((u[0], u[1] + 1)))
        start = cells[rng.integers(len(cells))]
        new.update_path(rel(start), rel(goal))
        ref.update_path(rel(start), rel(goal))
        assert_same_outputs(new, ref)
    assert not np.any(np.array(new.get_flow_matrix()))


def _cpp():
    from test_cpp_inference import _cpp_follower
    return _cpp_follower()


def _env(map_name, n, seed=3, steps=40):
    from env.create_env import create_env_base
    from follower.training_config import EnvironmentMazes
    env_cfg = EnvironmentMazes()
    env_cfg.grid_config.num_agents = n
    env_cfg.grid_config.map_name = map_name
    env_cfg.grid_config.max_episode_steps = steps
    env_cfg.grid_config.seed = seed
    return create_env_base(env_cfg)


def _config(Config, radius, table, **dflow):
    cfg = Config()
    cfg.obs_radius = radius
    cfg.num_threads = 1
    cfg.path_to_weights = ''
    cfg.use_memory = True
    cfg.neighbour_memory = 'trail'
    cfg.crisscross_ratio = 2.0
    cfg.crisscross_all = True
    if table is not None:
        cfg.guidance_table = table.ravel().tolist()
    for k, v in dflow.items():
        setattr(cfg, k, v)
    return cfg


def _guidance_like_table(shape, seed=0):
    """Contraflow-like table: factors 1 or in [1.04, 8], as the ggheur_zs tables."""
    rng = np.random.default_rng(seed)
    t = np.where(rng.random(shape + (4,)) < 0.5, 1.0, rng.uniform(1.04, 8.0, shape + (4,)))
    return t.astype(np.float32)


@pytest.mark.parametrize('map_name', [r'mazes-.+', r'wfi_warehouse'])
@pytest.mark.parametrize('use_table', [False, True])
def test_cpp_follower_dflow_off_equals_base(map_name, use_table):
    """dflow 'off' (explicitly, with arbitrary weights) gives exactly the observations of the untouched Config; 'add'
    on the same episodes changes some of them (so the comparison is not vacuous)."""
    Follower, Config = _cpp()
    n, radius = 48, 5
    env = _env(map_name, n)
    rng = np.random.default_rng(2)
    num_diff = 0
    for episode in range(2):
        obs, _ = env.reset(seed=episode)
        grid = np.array(env.get_global_obstacles(), dtype=np.float64)
        table = _guidance_like_table(grid.shape) if use_table else None
        algos = []
        for dflow in ({}, {'dflow': 'off', 'dflow_with': 0.0, 'dflow_opp': 2.0}, {'dflow': 'add'}):
            algos.append(Follower())
            algos[-1].init(_config(Config, env.grid_config.obs_radius, table, **dflow), grid, env.get_global_agents_xy())
        for step in range(40):
            for a in algos:
                a.set_state([tuple(o['xy']) for o in obs], [tuple(o['target_xy']) for o in obs])
            for k in range(n):
                base, off, add = (np.array(a.observe(k, radius), dtype=np.float32) for a in algos)
                np.testing.assert_array_equal(off, base)
                num_diff += int(not np.array_equal(add, base))
            assert not np.any(np.array(algos[1].get_flow(0)))
            actions = rng.integers(0, 5, size=n).tolist()
            for a in algos:
                a.set_actions(actions)
            obs, _, done, trunc, _ = env.step(list(actions))
            if all(done) or all(trunc):
                break
    assert num_diff > 0, 'dflow add never changed a path, the test does not exercise it'


# ---------------------------------------------------------------------------------------------------------------- 2.
def test_planner_flow_accumulation():
    grid = GRIDS['maze31'](0)
    offset = (1, 2)
    p = new_planner(grid.tolist(), True, True, True)
    p.set_abs_start(offset)
    p.set_dflow(1, 1.0, 1.0)
    cells = free_cells(grid)
    rel = lambda c: (c[0] - offset[0], c[1] - offset[1])
    expected = np.zeros(grid.shape + (4,))
    rng = np.random.default_rng(0)
    num = 0
    while num < 50:
        u = cells[rng.integers(len(cells))]
        k = int(rng.integers(4))
        v = (u[0] + DIRECTIONS[k][0], u[1] + DIRECTIONS[k][1])
        if grid[v]:
            continue
        p.add_observed_move(rel(u), rel(v))
        expected[u + (k,)] += 1
        expected[v + (k,)] += 1
        num += 1
    np.testing.assert_array_equal(np.array(p.get_flow_matrix()), expected)
    with pytest.raises(ValueError):
        p.add_observed_move(rel(cells[0]), rel((cells[0][0] + 1, cells[0][1] + 1)))
    with pytest.raises(ValueError):
        p.set_dflow(4, 1.0, 1.0)
    with pytest.raises(ValueError):
        p.set_dflow(2, -0.5, 1.0)


def _scene_follower(Follower, Config, size, starts, mode='add', radius=2):
    """Open size x size room with an obstacle border of `radius`; starts are absolute (row, col)."""
    grid = np.ones((size + 2 * radius, size + 2 * radius))
    grid[radius:-radius, radius:-radius] = 0
    cfg = Config()
    cfg.obs_radius = radius
    cfg.num_threads = 1
    cfg.path_to_weights = ''
    cfg.dflow = mode
    algo = Follower()
    algo.init(cfg, grid, [tuple(s) for s in starts])
    return algo, grid


def _set(algo, starts, positions, goals):
    rel = lambda c, s: (c[0] - s[0], c[1] - s[1])
    algo.set_state([rel(p, s) for p, s in zip(positions, starts)], [rel(g, s) for g, s in zip(goals, starts)])


E, S, N, W = 0, 1, 2, 3  # d4 indices


def test_follower_infers_moves_of_others():
    """Radius 3, observer A stays at (7, 7) and sees rows / columns 4..10 (absolute coordinates, as given to init).
    Step 1: B moves east (6, 5) -> (6, 6): credited to both cells. C stays at (9, 6): no flow. D moves west
    (8, 10) -> (8, 9) while E sits at (9, 9), a second candidate source of (8, 9): ambiguous, skipped.
    Step 2 (same goal): B moves east again (6, 6) -> (6, 7); A's own cell (7, 7), next to (6, 7), is no source; the
    flow accumulates. Step 3: A gets a new goal and C moves north (9, 6) -> (8, 6): the flow (and the occupations) are
    cleared first, then this step's move is added."""
    Follower, Config = _cpp()
    s = [(7, 7), (6, 5), (9, 6), (8, 10), (9, 9)]
    goals = [(14, 14)] * 5
    algo, grid = _scene_follower(Follower, Config, 13, s, radius=3)
    _set(algo, s, s, goals)
    algo.observe(0, 3)
    assert not np.any(np.array(algo.get_flow(0)))  # no previous view yet
    _set(algo, s, [(7, 7), (6, 6), (9, 6), (8, 9), (9, 9)], goals)
    algo.observe(0, 3)
    flow = np.array(algo.get_flow(0))
    expected = np.zeros_like(flow)
    expected[6, 5, E] = expected[6, 6, E] = 1
    np.testing.assert_array_equal(flow, expected)
    _set(algo, s, [(7, 7), (6, 7), (9, 6), (8, 9), (9, 9)], goals)
    algo.observe(0, 3)
    expected[6, 6, E] += 1
    expected[6, 7, E] += 1
    np.testing.assert_array_equal(np.array(algo.get_flow(0)), expected)
    assert np.array(algo.get_occupations(0)).sum() == 3 * 4
    _set(algo, s, [(7, 7), (6, 7), (8, 6), (8, 9), (9, 9)], [(13, 14)] + goals[1:])
    algo.observe(0, 3)
    flow = np.array(algo.get_flow(0))
    expected = np.zeros_like(flow)
    expected[9, 6, N] = expected[8, 6, N] = 1
    np.testing.assert_array_equal(flow, expected)
    assert np.array(algo.get_occupations(0)).sum() == 4  # only this step's 4 visible others: reset happened


def test_follower_skips_unseen_sources_and_own_cell():
    """Radius 2, A at (6, 6) sees rows / columns 4..8. Step 1: B moves west (6, 9) -> (6, 8); its source (6, 9) was
    outside A's view, so the arrival is not credited. Step 2: A moves west to (6, 5) and C enters A's previous cell
    from (5, 6) (south): A's own previous cell is not taken for an occupied source, so C's move is credited."""
    Follower, Config = _cpp()
    s = [(6, 6), (6, 9), (5, 6)]
    goals = [(10, 10)] * 3
    algo, _ = _scene_follower(Follower, Config, 11, s)
    _set(algo, s, s, goals)
    algo.observe(0, 2)
    _set(algo, s, [(6, 6), (6, 8), (5, 6)], goals)
    algo.observe(0, 2)
    assert not np.any(np.array(algo.get_flow(0)))
    _set(algo, s, [(6, 5), (6, 8), (6, 6)], goals)
    algo.observe(0, 2)
    flow = np.array(algo.get_flow(0))
    expected = np.zeros_like(flow)
    expected[5, 6, S] = expected[6, 6, S] = 1
    np.testing.assert_array_equal(flow, expected)


def test_flow_kept_without_reset_flag():
    """With reset_dynamic_cost off the occupations are never cleared, and neither is the flow."""
    Follower, Config = _cpp()
    s = [(4, 4), (4, 5)]
    algo, grid = _scene_follower(Follower, Config, 9, s)
    algo2 = Follower()
    cfg = Config()
    cfg.obs_radius = 2
    cfg.num_threads = 1
    cfg.path_to_weights = ''
    cfg.dflow = 'add'
    cfg.reset_dynamic_cost = False
    algo2.init(cfg, grid, s)
    for a in (algo, algo2):
        _set(a, s, s, [(8, 8)] * 2)
        a.observe(0, 2)
        _set(a, s, [(4, 4), (5, 5)], [(8, 8)] * 2)  # B moves south
        a.observe(0, 2)
        assert np.array(a.get_flow(0))[:, :, S].sum() == 2
        _set(a, s, [(4, 4), (5, 5)], [(3, 3), (8, 8)])  # A gets a new goal
        a.observe(0, 2)
    assert not np.any(np.array(algo.get_flow(0)))
    assert np.array(algo2.get_flow(0)).sum() == 2


def test_unknown_dflow_mode_rejected():
    Follower, Config = _cpp()
    cfg = Config()
    cfg.path_to_weights = ''
    cfg.dflow = 'sideways'
    with pytest.raises(ValueError, match='sideways'):
        Follower().init(cfg, np.pad(np.zeros((6, 6)), 5, constant_values=1), [(6, 6)])


@pytest.mark.skipif(os.environ.get('LTF_FIX_SOFT_POSITIONS') != '1',
                    reason="pogema's soft-collision bug hides agents, which makes stays look like moves")
@pytest.mark.parametrize('map_name', [r'mazes-.+', r'wfi_warehouse', r'pico_.+'])
def test_inferred_moves_are_true_moves(map_name):
    """On real episodes with random actions, every move added to any agent's flow is a true move u -> v of some agent
    on that step, and many moves are found."""
    Follower, Config = _cpp()
    n = 64 if map_name != r'pico_.+' else 16
    env = _env(map_name, n, seed=4, steps=60)
    rng = np.random.default_rng(5)
    num_inferred = 0
    obs, _ = env.reset(seed=0)
    grid = np.array(env.get_global_obstacles(), dtype=np.float64)
    starts = env.get_global_agents_xy()
    algo = Follower()
    algo.init(_config(Config, env.grid_config.obs_radius, None, dflow='add'), grid, starts)
    prev_abs, prev_goals = None, None
    for step in range(60):
        xy = [tuple(o['xy']) for o in obs]
        goals = [tuple(o['target_xy']) for o in obs]
        algo.set_state(xy, goals)
        cur_abs = [(x + s[0], y + s[1]) for (x, y), s in zip(xy, starts)]
        true_moves = set()
        if prev_abs is not None:
            true_moves = {(p, c) for p, c in zip(prev_abs, cur_abs) if p != c}
        for k in range(n):
            before = np.array(algo.get_flow(k))
            algo.observe(k, 5)
            diff = np.array(algo.get_flow(k)) - (before if prev_goals is None or prev_goals[k] == goals[k] else 0)
            assert diff.min() >= 0
            for i, j, d in zip(*np.nonzero(diff)):
                di, dj = DIRECTIONS[d]
                # the cell is either the source (move to the cell in direction d) or the target (move from behind)
                assert ((i, j), (i + di, j + dj)) in true_moves or ((i - di, j - dj), (i, j)) in true_moves, \
                    f'step {step} agent {k}: no agent moved through {(i, j)} in direction {(di, dj)}'
                num_inferred += 1
        prev_abs, prev_goals = cur_abs, goals
        actions = rng.integers(0, 5, size=n).tolist()
        algo.set_actions(actions)
        obs, _, done, trunc, _ = env.step(list(actions))
        if all(done) or all(trunc):
            break
    assert num_inferred > 50


# ---------------------------------------------------------------------------------------------------------------- 4.
@pytest.mark.parametrize('mode', ['add', 'split', 'splitsoft', 'ggboost'])
@pytest.mark.parametrize('cc_all', [False, True])
@pytest.mark.parametrize('grid_name', ['random20_d10', 'maze31', 'warehouse'])
def test_dflow_against_oracle(grid_name, cc_all, mode):
    """Random tables (factors >= 1), occupations and flows; without static costs the heuristic is admissible, so the
    path must be optimal for the directed costs of planner.h (f = table[u][k], o = opposite of k, occ / flow at v):
      add:     dyn = occ + flow[o]
      split:   dyn = max(0, occ + s ((w - 1) flow[k] + (x - 1) flow[o])), s = min(1, occ / sum(flow))
      ggboost: f' = 1 + (f - 1)(1 + flow[o]) if f > 1, dyn = occ
    cost = (1 + dyn) f' (crisscross_all) or f' + dyn."""
    rng = np.random.default_rng(23)
    grid = GRIDS[grid_name](2)
    cells = free_cells(grid)
    offset = (3, 1)
    rel = lambda c: (c[0] - offset[0], c[1] - offset[1])
    code, w_with, w_opp = {'add': (1, 1.0, 1.0), 'split': (2, 0.0, 2.0), 'splitsoft': (2, 0.5, 1.5),
                           'ggboost': (3, 1.0, 1.0)}[mode]
    num_checked = num_changed = 0
    for _ in range(25):
        table = np.where(rng.random(grid.shape + (4,)) < 0.5, 1.0,
                         rng.uniform(1.0, 4.0, grid.shape + (4,))).astype(np.float32)
        start, goal = cells[rng.integers(len(cells))], cells[rng.integers(len(cells))]
        planners = []
        for m in (code, 0):
            p = new_planner(grid.tolist(), False, True, True)
            p.set_abs_start(offset)
            p.set_crisscross(1.0, cc_all)
            p.set_guidance(table.ravel().tolist())
            p.set_dflow(m, w_with, w_opp)
            planners.append(p)
        occ = np.zeros(grid.shape)
        flow = np.zeros(grid.shape + (4,))
        # one window: the planner's goal is set by update_path only, so before the first plan every
        # update_occupations call with this goal resets (and the flow is added after it)
        # a window over the whole grid (top-left corner = absolute (0, 0)), so that occupations lie on the paths
        win = ((rng.random(grid.shape) < 0.3) & (grid == 0)).astype(np.float64)
        occ += win
        for p in planners:
            p.update_occupations(win, rel((0, 0)), rel(goal))
        for _ in range(4 * len(cells)):
            u = cells[rng.integers(len(cells))]
            k = int(rng.integers(4))
            v = (u[0] + DIRECTIONS[k][0], u[1] + DIRECTIONS[k][1])
            if grid[v]:
                continue
            for p in planners:
                p.add_observed_move(rel(u), rel(v))
            flow[u + (k,)] += 1
            flow[v + (k,)] += 1
        np.testing.assert_array_equal(np.array(planners[0].get_flow_matrix()), flow)
        p = planners[0]
        p.update_path(rel(start), rel(goal))
        planners[1].update_path(rel(start), rel(goal))
        path = [(a + offset[0], b + offset[1]) for a, b in p.get_path()]
        num_changed += int(p.get_path() != planners[1].get_path())

        def move_cost(u, v):
            k = DIRECTIONS.index((v[0] - u[0], v[1] - u[1]))
            o = 3 - k
            f = float(table[u[0], u[1], k])
            dyn = occ[v]
            fl = flow[v]
            if mode == 'add':
                dyn = dyn + fl[o]
            elif mode in ('split', 'splitsoft') and fl.sum() > 0:
                s = min(1.0, dyn / fl.sum())
                dyn = max(0.0, dyn + s * ((w_with - 1) * fl[k] + (w_opp - 1) * fl[o]))
            elif mode == 'ggboost' and f > 1:
                f = 1 + (f - 1) * (1 + fl[o])
            assert dyn >= 0 and f > 0
            return (1.0 + dyn) * f if cc_all else f + dyn

        best = directed_dijkstra_cost(grid, move_cost, start, goal)
        if start == goal or not np.isfinite(best):
            assert path == []
            continue
        assert path[0] == start and path[-1] == goal
        for a, b in zip(path, path[1:]):
            assert abs(a[0] - b[0]) + abs(a[1] - b[1]) == 1 and grid[b] == 0
        assert sum(move_cost(a, b) for a, b in zip(path, path[1:])) == pytest.approx(best, rel=1e-4)
        num_checked += 1
    assert num_checked > 15
    if grid_name != 'maze31':  # a perfect maze has one path between any two cells
        assert num_changed > 0, f'{mode} never changed a path'
