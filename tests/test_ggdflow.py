"""Density gate (Config.guidance_gate, planner::set_gate) and direction-aware dynamic cost (Config.dflow,
planner::set_dflow) together.

1. Planner oracle: with a gate in (0, 1) and dflow on, paths are optimal for the directed costs
   f = exp(log f_cc + g (log f_table - log f_cc)) (ggboost: f' = 1 + (f - 1)(1 + flow[o]) if f > 1) and
   dyn = occ (+ flow[o] for add, the split formula for split), cost = (1 + dyn) f' (crisscross_all) or f' + dyn;
   every edge cost of the grid stays >= 0 (dyn >= 0, f' > 0), also with extreme flows.
2. C++ Follower, real episodes: both flags on runs, the gates are exactly the gate-only Follower's (dflow does not touch
   the density), the flow is exactly the dflow-only Follower's (the gate does not touch the flow), and the paths differ
   from both parents on some steps. Each flag off (explicit 'off' / no gate) reproduces the other parent exactly.

Run: LTF_FIX_SOFT_POSITIONS=1 python -m pytest tests/test_ggdflow.py -v
"""
import numpy as np
import pytest

from planner_common import free_cells, new_planner
from test_planner import GRIDS, directed_dijkstra_cost
from follower_cpp.guidance import DIRECTIONS, crisscross_table

RATIO = 2.0
DFLOW = {'add': (1, 1.0, 1.0), 'split': (2, 0.0, 2.0), 'splitsoft': (2, 0.5, 1.5), 'splitmild': (2, 0.75, 1.25),
         'ggboost': (3, 1.0, 1.0)}


def _blend(grid, table, gate):
    cc = crisscross_table(grid.shape, RATIO).astype(np.float64)
    return np.exp(np.log(cc) + gate * (np.log(table.astype(np.float64)) - np.log(cc)))


def _cost_fn(mode, blend, occ, flow, cc_all, w_with, w_opp):
    def move_cost(u, v):
        k = DIRECTIONS.index((v[0] - u[0], v[1] - u[1]))
        o = 3 - k
        f = float(blend[u[0], u[1], k])
        dyn = float(occ[v])
        fl = flow[v]
        if mode == 'add':
            dyn = dyn + fl[o]
        elif mode.startswith('split') and fl.sum() > 0:
            s = min(1.0, dyn / fl.sum())
            dyn = max(0.0, dyn + s * ((w_with - 1) * fl[k] + (w_opp - 1) * fl[o]))
        elif mode == 'ggboost' and f > 1:
            f = 1 + (f - 1) * (1 + fl[o])
        assert dyn >= 0 and f > 0
        return (1.0 + dyn) * f if cc_all else f + dyn
    return move_cost


# ---------------------------------------------------------------------------------------------------------------- 1.
@pytest.mark.parametrize('mode', list(DFLOW))
@pytest.mark.parametrize('gate', [0.3, 0.7])
@pytest.mark.parametrize('cc_all', [False, True])
@pytest.mark.parametrize('grid_name', ['random20_d10', 'maze31', 'warehouse'])
def test_gate_and_dflow_against_oracle(grid_name, cc_all, gate, mode):
    rng = np.random.default_rng(31)
    grid = GRIDS[grid_name](2)
    cells = free_cells(grid)
    offset = (3, 1)
    rel = lambda c: (c[0] - offset[0], c[1] - offset[1])
    code, w_with, w_opp = DFLOW[mode]
    num_checked = num_changed = 0
    for t in range(20):
        # ggheur-like tables: factors 1 or > 1, plus some < 1 (h scaled by h_scale^gate)
        table = np.where(rng.random(grid.shape + (4,)) < 0.5, 1.0,
                         rng.uniform(0.5, 6.0, grid.shape + (4,))).astype(np.float32)
        blend = _blend(grid, table, gate)
        start, goal = cells[rng.integers(len(cells))], cells[rng.integers(len(cells))]
        planners = []
        for m in (code, 0):  # combined, gate only
            p = new_planner(grid.tolist(), False, True, True)
            p.set_abs_start(offset)
            p.set_crisscross(RATIO, cc_all)
            p.set_guidance(table.ravel().tolist())
            p.set_guidance_gate(True)
            p.set_gate(gate)
            p.set_dflow(m, w_with, w_opp)
            planners.append(p)
        assert planners[0].get_gate() == pytest.approx(gate)
        occ = np.zeros(grid.shape)
        flow = np.zeros(grid.shape + (4,))
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
        num_changed += int(p.get_path() != planners[1].get_path())
        move_cost = _cost_fn(mode, blend, occ, flow, cc_all, w_with, w_opp)
        # every edge of the grid is non-negative under the combined costs
        for u in cells:
            for d in DIRECTIONS:
                v = (u[0] + d[0], u[1] + d[1])
                if 0 <= v[0] < grid.shape[0] and 0 <= v[1] < grid.shape[1] and not grid[v]:
                    assert move_cost(u, v) >= 0
        path = [(a + offset[0], b + offset[1]) for a, b in p.get_path()]
        best = directed_dijkstra_cost(grid, move_cost, start, goal)
        if start == goal or not np.isfinite(best):
            assert path == []
            continue
        assert path[0] == start and path[-1] == goal
        for a, b in zip(path, path[1:]):
            assert abs(a[0] - b[0]) + abs(a[1] - b[1]) == 1 and grid[b] == 0
        assert sum(move_cost(a, b) for a, b in zip(path, path[1:])) == pytest.approx(best, rel=1e-4)
        num_checked += 1
    assert num_checked > 12
    # a perfect maze has one path between any two cells; splitmild (0.75 / 1.25) is too weak to change a path in 20
    # random trials on every grid, its effect is asserted on real episodes in test_cpp_follower_gate_and_dflow
    if grid_name != 'maze31' and mode != 'splitmild':
        assert num_changed > 0, f'{mode} never changed a gated path'


@pytest.mark.parametrize('w_with, w_opp', [(0.0, 2.0), (0.0, 0.0), (0.5, 1.5), (0.75, 1.25), (0.0, 10.0)])
def test_split_dyn_non_negative_extreme_flows(w_with, w_opp):
    """Split keeps dyn >= 0 for any occupation count and flow (also flows far above the occupations)."""
    rng = np.random.default_rng(0)
    for _ in range(20000):
        occ = float(rng.integers(0, 4))
        fl = rng.integers(0, 50, size=4).astype(float) * (rng.random(4) < 0.7)
        k = int(rng.integers(4))
        dyn = occ
        if fl.sum() > 0:
            s = min(1.0, occ / fl.sum())
            dyn = max(0.0, occ + s * ((w_with - 1) * fl[k] + (w_opp - 1) * fl[3 - k]))
        assert dyn >= 0
        assert dyn >= occ * min(w_with, w_opp, 1.0) - 1e-9  # the split never removes more than the weights allow


# ---------------------------------------------------------------------------------------------------------------- 2.
def _env(map_name, n, seed=3, steps=40):
    from env.create_env import create_env_base
    from follower.training_config import EnvironmentMazes
    env_cfg = EnvironmentMazes()
    env_cfg.grid_config.num_agents = n
    env_cfg.grid_config.map_name = map_name
    env_cfg.grid_config.max_episode_steps = steps
    env_cfg.grid_config.seed = seed
    return create_env_base(env_cfg)


def _config(Config, radius, table, gate=None, **dflow):
    cfg = Config()
    cfg.obs_radius = radius
    cfg.num_threads = 1
    cfg.path_to_weights = ''
    cfg.use_memory = True
    cfg.neighbour_memory = 'trail'
    cfg.crisscross_ratio = RATIO
    cfg.crisscross_all = True
    cfg.guidance_table = table.ravel().tolist()
    if gate is not None:
        cfg.guidance_gate = True
        cfg.gate_rho0, cfg.gate_rho1 = gate
        cfg.gate_halflife = 0.0  # raw rho, as ggdraw
    for k, v in dflow.items():
        setattr(cfg, k, v)
    return cfg


@pytest.mark.parametrize('dflow', ['split', 'splitmild', 'ggboost'])
@pytest.mark.parametrize('map_name', [r'mazes-.+', r'wfi_warehouse'])
def test_cpp_follower_gate_and_dflow(map_name, dflow):
    from test_cpp_inference import _cpp_follower
    Follower, Config = _cpp_follower()
    n, radius = 96, 5
    env = _env(map_name, n)
    rng = np.random.default_rng(7)
    code, w_with, w_opp = DFLOW[dflow]
    dflow_cfg = {'dflow': 'split' if dflow.startswith('split') else dflow, 'dflow_with': w_with, 'dflow_opp': w_opp}
    gate = (0.02, 0.07)
    num_vs_gate = num_vs_dflow = 0
    num_mid_gates = 0
    for episode in range(2):
        obs, _ = env.reset(seed=episode)
        grid = np.array(env.get_global_obstacles(), dtype=np.float64)
        table = np.where(rng.random(grid.shape + (4,)) < 0.5, 1.0,
                         rng.uniform(1.04, 8.0, grid.shape + (4,))).astype(np.float32)
        cfgs = {
            'both': _config(Config, radius, table, gate, **dflow_cfg),
            'gate': _config(Config, radius, table, gate),
            'gate_dflow_off': _config(Config, radius, table, gate, dflow='off', dflow_with=0.0, dflow_opp=2.0),
            'dflow': _config(Config, radius, table, None, **dflow_cfg),
            'dflow_gate_off': _config(Config, radius, table, None, **dflow_cfg),
        }
        cfgs['dflow_gate_off'].guidance_gate = False
        algos = {}
        for name, cfg in cfgs.items():
            algos[name] = Follower()
            algos[name].init(cfg, grid, env.get_global_agents_xy())
        for step in range(40):
            for a in algos.values():
                a.set_state([tuple(o['xy']) for o in obs], [tuple(o['target_xy']) for o in obs])
            for k in range(n):
                out = {name: np.array(a.observe(k, radius), dtype=np.float32) for name, a in algos.items()}
                np.testing.assert_array_equal(out['gate_dflow_off'], out['gate'])
                np.testing.assert_array_equal(out['dflow_gate_off'], out['dflow'])
                np.testing.assert_array_equal(np.array(algos['both'].get_flow(k)), np.array(algos['dflow'].get_flow(k)))
                assert np.all(np.array(algos['both'].get_occupations(k)) >= 0)
                num_vs_gate += int(not np.array_equal(out['both'], out['gate']))
                num_vs_dflow += int(not np.array_equal(out['both'], out['dflow']))
            g_both, g_gate = np.array(algos['both'].get_gates()), np.array(algos['gate'].get_gates())
            np.testing.assert_array_equal(g_both, g_gate)
            assert np.all((g_both >= 0) & (g_both <= 1))
            num_mid_gates += int(np.sum((g_both > 0) & (g_both < 1)))
            assert set(algos['dflow'].get_gates()) == {1.0}  # no gate: the plain table
            actions = rng.integers(0, 5, size=n).tolist()
            for a in algos.values():
                a.set_actions(actions)
            obs, _, done, trunc, _ = env.step(list(actions))
            if all(done) or all(trunc):
                break
    assert num_mid_gates > 0, 'no gate strictly between 0 and 1: the blend is not exercised'
    assert num_vs_gate > 0, 'dflow never changed a gated path'
    assert num_vs_dflow > 0, 'the gate never changed a dflow path'
