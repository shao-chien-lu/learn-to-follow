"""Density gate of the guidance tables (planner::set_guidance_gate / set_gate, Config.guidance_gate).

1. Planner: gate 0 is bit-identical to no table (crisscross as set), gate 1 to the plain table; in between paths are
   optimal for the log-space blend of the crisscross and the table factors.
2. C++ Follower: guidance_gate with g == 0 everywhere (rho0 above any density) gives the same observations (A* paths)
   as ctrl (crisscross 2 / all, no table); with g == 1 everywhere (rho1 below any density) the same as the plain table.
3. The gate's density is the agent's own: other agents in its obs_radius window / free cells there (raw, half-life 0),
   and the EMA follows rho_t = rho_{t-1} + (1 - 2^(-1/h)) (rho - rho_{t-1}).

Run: python -m pytest tests/test_guidance_gate.py -v
"""
import numpy as np
import pytest

from planner_common import free_cells, new_planner, occupation_window
from test_planner import FLAGS, GRIDS, directed_dijkstra_cost, make_pair
from test_guidance import _episode
from follower_cpp.guidance import DIRECTIONS, crisscross_table

RATIO = 2.0


def _random_table(grid, seed, lo=0.5, hi=8.0):
    rng = np.random.default_rng(seed)
    return np.exp(rng.uniform(np.log(lo), np.log(hi), size=grid.shape + (4,))).astype(np.float32)


@pytest.mark.parametrize('flags', FLAGS, ids=lambda f: 'S%dD%dR%d' % tuple(map(int, f)))
@pytest.mark.parametrize('cc_all', [False, True])
@pytest.mark.parametrize('grid_name', ['maze31', 'warehouse', 'random20_d30'])
def test_gate_zero_equals_crisscross(grid_name, cc_all, flags):
    grid = GRIDS[grid_name](1)
    offset = (2, 1)
    gated, ctrl = make_pair(grid, flags, offset)[0], make_pair(grid, flags, offset)[0]
    for p in (gated, ctrl):
        p.set_crisscross(RATIO, cc_all)
    gated.set_guidance(_random_table(grid, 3).ravel().tolist())
    gated.set_guidance_gate(True)
    gated.set_gate(0.0)
    _episode(gated, ctrl, grid, offset, seed=7)
    gated.set_gate(-0.5)  # clipped to 0
    assert gated.get_gate() == 0.0
    _episode(gated, ctrl, grid, offset, seed=8)


@pytest.mark.parametrize('flags', FLAGS, ids=lambda f: 'S%dD%dR%d' % tuple(map(int, f)))
@pytest.mark.parametrize('cc_all', [False, True])
@pytest.mark.parametrize('grid_name', ['maze31', 'warehouse', 'random20_d30'])
def test_gate_one_equals_table(grid_name, cc_all, flags):
    grid = GRIDS[grid_name](1)
    offset = (2, 1)
    table = _random_table(grid, 4).ravel().tolist()
    gated, plain = make_pair(grid, flags, offset)[0], make_pair(grid, flags, offset)[0]
    for p in (gated, plain):
        p.set_crisscross(RATIO, cc_all)
        p.set_guidance(table)
    gated.set_guidance_gate(True)
    gated.set_gate(1.0)
    _episode(gated, plain, grid, offset, seed=7)
    gated.set_gate(3.0)  # clipped to 1
    assert gated.get_gate() == 1.0
    _episode(gated, plain, grid, offset, seed=8)


def test_gate_off_by_default_and_needs_table():
    grid = GRIDS['maze31'](0)
    p = new_planner(grid.tolist(), True, True, True)
    assert p.get_gate() == 1.0
    with pytest.raises(ValueError):
        p.set_guidance_gate(True)  # no table
    p.set_guidance(_random_table(grid, 1).ravel().tolist())
    p.set_guidance_gate(True)
    with pytest.raises(ValueError):
        p.set_gate(float('nan'))


@pytest.mark.parametrize('gate', [0.25, 0.5, 0.8])
@pytest.mark.parametrize('use_dynamic, cc_all', [(False, False), (True, False), (True, True)])
@pytest.mark.parametrize('grid_name', ['random20_d10', 'maze31', 'warehouse'])
def test_blend_against_oracle(grid_name, use_dynamic, cc_all, gate):
    """Paths are optimal for the factors exp(log cc + g (log table - log cc)) (no static costs: admissible h)."""
    rng = np.random.default_rng(23)
    grid = GRIDS[grid_name](2)
    cells = free_cells(grid)
    offset = (3, 1)
    rel = lambda c: (c[0] - offset[0], c[1] - offset[1])
    cc = crisscross_table(grid.shape, RATIO).astype(np.float64)
    num_checked = 0
    for t in range(30):
        table = _random_table(grid, 100 + t, 0.25, 8.0)
        blend = np.exp(np.log(cc) + gate * (np.log(table.astype(np.float64)) - np.log(cc)))
        start, goal = cells[rng.integers(len(cells))], cells[rng.integers(len(cells))]
        win, cur = occupation_window(rng, grid, offset)
        occ = np.zeros(grid.shape)
        ii, jj = np.nonzero(win)
        occ[ii + cur[0] + offset[0], jj + cur[1] + offset[1]] += 1

        p = new_planner(grid.tolist(), False, use_dynamic, True)
        p.set_abs_start(offset)
        p.set_crisscross(RATIO, cc_all)
        p.set_guidance(table.ravel().tolist())
        p.set_guidance_gate(True)
        p.set_gate(gate)
        p.update_occupations(win, cur, rel(goal))
        p.update_path(rel(start), rel(goal))
        path = [(a + offset[0], b + offset[1]) for a, b in p.get_path()]

        def move_cost(u, v):
            factor = float(blend[u[0], u[1], DIRECTIONS.index((v[0] - u[0], v[1] - u[1]))])
            dyn = occ[v] if use_dynamic else 0.0
            return (1.0 + dyn) * factor if cc_all else factor + dyn

        best = directed_dijkstra_cost(grid, move_cost, start, goal)
        if start == goal or not np.isfinite(best):
            assert path == []
            continue
        assert path[0] == start and path[-1] == goal
        assert sum(move_cost(a, b) for a, b in zip(path, path[1:])) == pytest.approx(best, rel=1e-4)
        num_checked += 1
    assert num_checked > 20


def _env_and_follower(map_name, n, seed=3, steps=40):
    from test_cpp_inference import _cpp_follower
    from env.create_env import create_env_base
    from follower.training_config import EnvironmentMazes
    from follower.preprocessing import PreprocessorConfig, wrap_preprocessors
    Follower, Config = _cpp_follower()
    env_cfg = EnvironmentMazes()
    env_cfg.grid_config.num_agents = n
    env_cfg.grid_config.map_name = map_name
    env_cfg.grid_config.max_episode_steps = steps
    env_cfg.grid_config.seed = seed
    env = wrap_preprocessors(create_env_base(env_cfg), PreprocessorConfig(
        network_input_radius=5, use_memory=True, neighbour_memory='trail', crisscross_ratio=RATIO,
        crisscross_all=True))
    return env, Follower, Config


def _config(Config, env, table=None, gate=None, halflife=4.0):
    cfg = Config()
    cfg.obs_radius = env.grid_config.obs_radius
    cfg.num_threads = 1
    cfg.path_to_weights = ''
    cfg.use_memory = True
    cfg.neighbour_memory = 'trail'
    cfg.crisscross_ratio = RATIO
    cfg.crisscross_all = True
    if table is not None:
        cfg.guidance_table = table.ravel().tolist()
    if gate is not None:
        cfg.guidance_gate = True
        cfg.gate_rho0, cfg.gate_rho1 = gate
        cfg.gate_halflife = halflife
    return cfg


@pytest.mark.parametrize('map_name', [r'mazes-.+', r'wfi_warehouse'])
def test_cpp_follower_gate_identities(map_name):
    """g == 0 (rho0 = 2 > any density) == ctrl; g == 1 (rho1 = -1 < any density) == plain table; over episodes with
    random actions, every agent's observation (A* path drawn in) compared."""
    n, radius = 64, 5
    env, Follower, Config = _env_and_follower(map_name, n)
    rng = np.random.default_rng(2)
    num_differ = 0
    for episode in range(2):
        obs, _ = env.reset(seed=episode)
        grid = np.array(env.get_global_obstacles(), dtype=np.float64)
        table = _random_table(grid, 10 + episode)
        cfgs = {'ctrl': _config(Config, env), 'g0': _config(Config, env, table, (2.0, 3.0)),
                'table': _config(Config, env, table), 'g1': _config(Config, env, table, (-2.0, -1.0))}
        algos = {}
        for k, cfg in cfgs.items():
            algos[k] = Follower()
            algos[k].init(cfg, grid, env.get_global_agents_xy())
        for step in range(40):
            for a in algos.values():
                a.set_state([tuple(o['xy']) for o in obs], [tuple(o['target_xy']) for o in obs])
            for k in range(n):
                out = {name: np.array(a.observe(k, radius), dtype=np.float32) for name, a in algos.items()}
                np.testing.assert_array_equal(out['g0'], out['ctrl'])
                np.testing.assert_array_equal(out['g1'], out['table'])
                num_differ += int(not np.array_equal(out['ctrl'], out['table']))
            assert set(algos['g0'].get_gates()) == {0.0}
            assert set(algos['g1'].get_gates()) == {1.0}
            actions = rng.integers(0, 5, size=n).tolist()
            for a in algos.values():
                a.set_actions(actions)
            obs, _, done, trunc, _ = env.step(list(actions))
            if all(done) or all(trunc):
                break
    assert num_differ > 0  # the table changes paths, so the identities are not vacuous


def test_cpp_follower_density_is_own_window():
    """rho0 = 0, rho1 = 1, raw: the gate equals the agent's local density (other agents in its 11 x 11 window / free
    cells of the window other than its own); with half-life h the gate follows the EMA of that density."""
    n, radius, h = 128, 5, 4.0
    env, Follower, Config = _env_and_follower(r'mazes-.+', n, seed=5)
    rng = np.random.default_rng(4)
    obs, _ = env.reset(seed=1)
    grid = np.array(env.get_global_obstacles(), dtype=np.float64)
    table = _random_table(grid, 1)
    raw, ema = Follower(), Follower()
    raw.init(_config(Config, env, table, (0.0, 1.0), halflife=0.0), grid, env.get_global_agents_xy())
    ema.init(_config(Config, env, table, (0.0, 1.0), halflife=h), grid, env.get_global_agents_xy())
    r = env.grid_config.obs_radius
    alpha = 1 - 2 ** (-1 / h)
    expected_ema = None
    num_nonzero = 0
    for step in range(30):
        xy = [tuple(o['xy']) for o in obs]
        for a in (raw, ema):
            a.set_state(xy, [tuple(o['target_xy']) for o in obs])
            for k in range(n):
                a.observe(k, radius)
        pos = np.array(env.get_global_agents_xy())
        occupied = np.zeros(grid.shape, bool)
        occupied[pos[:, 0], pos[:, 1]] = True
        rho = np.zeros(n)
        for k, (i, j) in enumerate(pos):
            win_free = grid[i - r:i + r + 1, j - r:j + r + 1] == 0
            win_occ = occupied[i - r:i + r + 1, j - r:j + r + 1]
            rho[k] = (win_occ.sum() - 1) / (win_free.sum() - 1)
        np.testing.assert_allclose(raw.get_gates(), np.clip(rho, 0, 1), rtol=1e-6, atol=1e-7)
        expected_ema = rho.copy() if expected_ema is None else expected_ema + alpha * (rho - expected_ema)
        np.testing.assert_allclose(ema.get_gates(), np.clip(expected_ema, 0, 1), rtol=1e-5, atol=1e-6)
        num_nonzero += int(np.count_nonzero(rho))
        actions = rng.integers(0, 5, size=n).tolist()
        for a in (raw, ema):
            a.set_actions(actions)
        obs, _, done, trunc, _ = env.step(list(actions))
    assert num_nonzero > 0
