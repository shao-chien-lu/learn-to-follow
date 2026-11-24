"""The python planner path (BatchPlanner -> multi_planner, used by the Follower via follower_preprocessor) with the
eval-only options of the C++ Follower (guidance table, density gate, dflow) must give exactly the C++ Follower's
observations (A* path in the obstacle channel), gates, flow and occupation counts, on identically seeded pogema envs
with random actions; and follower_preprocessor must pass FollowerInferenceConfig's options to the planner.

Run: LTF_FIX_SOFT_POSITIONS=1 python -m pytest tests/test_python_ggdflow.py -v
"""
import numpy as np
import pytest

import planner_common  # noqa: F401  (puts the repo root and tests/reference on sys.path)
from env.create_env import create_env_base
from follower.planning import DFLOW_MODES
from follower.preprocessing import FollowerWrapper, PreprocessorConfig, wrap_preprocessors
from follower.training_config import EnvironmentMazes
from follower_cpp.guidance import save_table
from test_cpp_inference import _cpp_follower

RADIUS = 5  # Follower: network_input_radius = obs_radius = 5

VARIANTS = {
    # the ggdflow_split arm: crisscross 2 all + raw gate + split 0/2
    'ggdflow_split': dict(crisscross_ratio=2.0, crisscross_all=True, table=True, guidance_gate_rho0=0.02,
                          guidance_gate_rho1=0.07, guidance_gate_halflife=0.0, dflow='split', dflow_with=0.0,
                          dflow_opp=2.0),
    'gate_ema_add': dict(crisscross_ratio=1.5, crisscross_all=False, table=True, guidance_gate_rho0=0.0,
                         guidance_gate_rho1=0.1, guidance_gate_halflife=4.0, dflow='add'),
    'table_ggboost': dict(crisscross_ratio=2.0, crisscross_all=True, table=True, dflow='ggboost'),
    'gate_only': dict(crisscross_ratio=2.0, crisscross_all=True, table=True, guidance_gate_rho0=0.02,
                      guidance_gate_rho1=0.07, guidance_gate_halflife=0.0),
    'split_only': dict(crisscross_ratio=2.0, crisscross_all=True, table=False, dflow='split', dflow_with=0.5,
                       dflow_opp=1.5),
    'cc2all': dict(crisscross_ratio=2.0, crisscross_all=True, table=False),
}


def _table(shape, seed):
    """Contraflow-like table (as the ggheur_zs tables): factors 1 or in [1.04, 8]."""
    rng = np.random.default_rng(seed)
    t = np.where(rng.random(shape + (4,)) < 0.5, 1.0, rng.uniform(1.04, 8.0, shape + (4,)))
    return t.astype(np.float32)


def _follower_wrapper(env):
    while not isinstance(env, FollowerWrapper):
        env = env.env
    return env


@pytest.mark.parametrize('variant', list(VARIANTS))
@pytest.mark.parametrize('map_name, n', [(r'mazes-.+', 48), (r'wfi_warehouse', 64), (r'pico_.+', 24)])
def test_python_planner_equals_cpp_follower(tmp_path, map_name, n, variant):
    Follower, Config = _cpp_follower()
    v = dict(VARIANTS[variant])
    use_table = v.pop('table')
    env_cfg = EnvironmentMazes()
    env_cfg.grid_config.num_agents = n
    env_cfg.grid_config.map_name = map_name
    env_cfg.grid_config.max_episode_steps = 60
    env_cfg.grid_config.seed = 5
    guidance_dir = str(tmp_path / 'tables') if use_table else None
    env = wrap_preprocessors(create_env_base(env_cfg),
                             PreprocessorConfig(network_input_radius=RADIUS, guidance_dir=guidance_dir, **v))
    probe = create_env_base(env_cfg)  # the same map, to store its table before the planner loads it on reset
    rng = np.random.default_rng(1)
    num_gated = num_flow = 0
    for episode in range(2):
        probe.reset(seed=episode)
        table = None
        if use_table:
            table = _table(np.asarray(probe.get_global_obstacles()).shape, episode)
            save_table(guidance_dir, probe.get_global_obstacles(), n, table)
        obs, _ = env.reset(seed=episode)
        glob = np.array(env.get_global_obstacles(), dtype=np.float64)
        np.testing.assert_array_equal(glob, np.asarray(probe.get_global_obstacles()))
        mp = _follower_wrapper(env).planner.planner
        cfg = Config()
        cfg.obs_radius = env.grid_config.obs_radius
        cfg.num_threads = 1
        cfg.path_to_weights = ''
        cfg.crisscross_ratio = v['crisscross_ratio']
        cfg.crisscross_all = v['crisscross_all']
        if use_table:
            cfg.guidance_table = table.ravel().tolist()
        if 'guidance_gate_rho0' in v:
            cfg.guidance_gate = True
            cfg.gate_rho0 = v['guidance_gate_rho0']
            cfg.gate_rho1 = v['guidance_gate_rho1']
            cfg.gate_halflife = v['guidance_gate_halflife']
        for key in ['dflow', 'dflow_with', 'dflow_opp']:
            if key in v:
                setattr(cfg, key, v[key])
        algo = Follower()
        algo.init(cfg, glob, env.get_global_agents_xy())
        for step in range(60):
            algo.set_state([tuple(o['xy']) for o in obs], [tuple(o['target_xy']) for o in obs])
            for k, o in enumerate(obs):
                cpp = np.array(algo.observe(k, RADIUS), dtype=np.float32)
                np.testing.assert_array_equal(cpp, o['obs'].ravel(), err_msg=f'ep {episode} step {step} agent {k}')
            if 'guidance_gate_rho0' in v:
                g_py, g_cpp = np.array(mp.get_gates()), np.array(algo.get_gates())
                np.testing.assert_array_equal(g_py, g_cpp, err_msg=f'gates, ep {episode} step {step}')
                num_gated += int(np.sum((g_py > 0) & (g_py < 1)))
            if step % 10 == 9:
                for k in range(n):
                    np.testing.assert_array_equal(np.array(mp.get_occupations(k)), np.array(algo.get_occupations(k)))
                    f_py, f_cpp = np.array(mp.get_flow(k)), np.array(algo.get_flow(k))
                    np.testing.assert_array_equal(f_py, f_cpp, err_msg=f'flow, ep {episode} step {step} agent {k}')
                    num_flow += int(f_py.sum())
            actions = rng.integers(0, 5, size=n).tolist()
            algo.set_actions(actions)
            obs, _, done, trunc, _ = env.step(list(actions))
            if all(done) or all(trunc):
                break
    if 'guidance_gate_rho0' in v and map_name != r'pico_.+':
        assert num_gated > 0, 'no gate strictly between 0 and 1: the blend was not exercised'
    if v.get('dflow', 'off') != 'off':
        assert num_flow > 0, 'no flow was inferred: dflow was not exercised'


def test_follower_preprocessor_passes_eval_options(tmp_path):
    """FollowerInferenceConfig's eval-time options reach the wrapper's PreprocessorConfig (and nothing changes without
    them), so they cannot be dropped silently."""
    from types import SimpleNamespace
    from follower.inference import FollowerInferenceConfig
    from follower.preprocessing import EVAL_PLANNER_KEYS, follower_preprocessor
    env_cfg = EnvironmentMazes()
    env_cfg.grid_config.num_agents = 8
    env_cfg.grid_config.map_name = r'pico_.+'
    base = PreprocessorConfig(network_input_radius=RADIUS)
    fields = FollowerInferenceConfig.__fields__
    assert all(k in fields for k in EVAL_PLANNER_KEYS)

    def algo(**kw):
        a = {k: fields[k].default for k in EVAL_PLANNER_KEYS}
        a.update(kw)
        return SimpleNamespace(training_config=SimpleNamespace(preprocessing=base), **a)

    env = follower_preprocessor(create_env_base(env_cfg), algo())
    assert _follower_wrapper(env)._cfg == base
    opts = dict(crisscross_ratio=2.0, crisscross_all=True, guidance_dir=str(tmp_path), guidance_gate_rho0=0.02,
                guidance_gate_rho1=0.07, guidance_gate_halflife=0.0, dflow='split', dflow_with=0.0, dflow_opp=2.0)
    env = follower_preprocessor(create_env_base(env_cfg), algo(**opts))
    cfg = _follower_wrapper(env)._cfg
    for k, val in opts.items():
        assert getattr(cfg, k) == val, k
    assert cfg.network_input_radius == RADIUS
    assert DFLOW_MODES[cfg.dflow] == 2
