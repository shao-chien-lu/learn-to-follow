import os
from typing import Optional

import numpy as np
from pogema import GridConfig

# noinspection PyUnresolvedReferences
import cppimport.import_hook
# noinspection PyUnresolvedReferences
from follower_cpp.planner import planner, multi_planner

from pydantic import BaseModel

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal


class PlannerConfig(BaseModel):
    use_static_cost: bool = True
    use_dynamic_cost: bool = True
    reset_dynamic_cost: bool = True
    # soft crisscross highways: moves against them cost crisscross_ratio times as much (1: off); with crisscross_all
    # the factor also scales the dynamic cost; see planner.h
    crisscross_ratio: float = 1.0
    crisscross_all: bool = False
    # Eval-only options of the C++ Follower's planner (follower_cpp/inference.py FollowerConfigCPP), here for the python
    # path (BatchPlanner / multi_planner); the defaults keep the planner unchanged.
    # guidance_dir: per-map tables <guidance_key(global obstacles, num_agents)>.npy (follower_cpp/guidance.py)
    guidance_dir: Optional[str] = None
    # density gate of the tables: clip((rho - rho0) / (rho1 - rho0), 0, 1), rho = other agents / free cells of the
    # agent's own observation window, EMA with half-life guidance_gate_halflife (<= 0: raw)
    guidance_gate_rho0: Optional[float] = None
    guidance_gate_rho1: Optional[float] = None
    guidance_gate_halflife: float = 4.0
    # direction-aware dynamic cost from the observed flow of other agents (planner::set_dflow)
    dflow: Literal['off', 'add', 'split', 'ggboost'] = 'off'
    dflow_with: float = 1.0
    dflow_opp: float = 1.0


DFLOW_MODES = {'off': 0, 'add': 1, 'split': 2, 'ggboost': 3}
# one line per distinct planner config and process, as evidence of the settings that reached the planner
_CONFIGS_DUMPED = set()


# Static penalties depend only on the map and obs radius; training cycles over a fixed set of maps,
# so cache them instead of re-running the O(cells^2) all-pairs BFS on every episode reset.
_PENALTY_CACHE = {}
_PENALTY_CACHE_MAX = 256


def get_penalties(obstacles, obs_radius, cfg):
    key = (np.asarray(obstacles, dtype=np.int8).tobytes(), len(obstacles), obs_radius)
    if key not in _PENALTY_CACHE:
        if len(_PENALTY_CACHE) >= _PENALTY_CACHE_MAX:
            _PENALTY_CACHE.pop(next(iter(_PENALTY_CACHE)))
        pen_calc = planner(obstacles, cfg.use_static_cost, cfg.use_dynamic_cost, cfg.reset_dynamic_cost)
        _PENALTY_CACHE[key] = pen_calc.precompute_penalty_matrix(obs_radius)
    return _PENALTY_CACHE[key]


class Planner:
    def __init__(self, cfg: PlannerConfig):
        self.planner = None
        self.obstacles = None
        self.starts = None
        self.cfg = cfg

    def add_grid_obstacles(self, obstacles, starts):
        self.obstacles = obstacles
        self.starts = starts
        self.planner = None

    def update(self, obs):
        num_agents = len(obs)
        obs_radius = len(obs[0]['obstacles']) // 2
        if self.planner is None:
            self.planner = [planner(self.obstacles, self.cfg.use_static_cost, self.cfg.use_dynamic_cost, self.cfg.reset_dynamic_cost) for _ in range(num_agents)]
            for i, p in enumerate(self.planner):
                p.set_abs_start(self.starts[i])
                p.set_crisscross(self.cfg.crisscross_ratio, self.cfg.crisscross_all)
            if self.cfg.use_static_cost:
                penalties = get_penalties(self.obstacles, obs_radius, self.cfg)
                for p in self.planner:
                    p.set_penalties(penalties)

        for k in range(num_agents):
            if obs[k]['xy'] == obs[k]['target_xy']:
                continue
            obs[k]['agents'][obs_radius][obs_radius] = 0
            self.planner[k].update_occupations(obs[k]['agents'], (obs[k]['xy'][0] - obs_radius, obs[k]['xy'][1] - obs_radius), obs[k]['target_xy'])
            obs[k]['agents'][obs_radius][obs_radius] = 1
            self.planner[k].update_path(obs[k]['xy'], obs[k]['target_xy'])

    def get_path(self):
        results = []
        for idx in range(len(self.planner)):
            results.append(self.planner[idx].get_path())
        return results


class BatchPlanner:
    """All agents of one env planned by a single native call per step (see multi_planner in planner.h).

    step() re-plans every agent, draws the path into each agent's obstacle channel (in place) and returns the
    intrinsic rewards; it replaces Planner + the per-agent post-processing loop of FollowerWrapper.
    """

    def __init__(self, cfg: PlannerConfig, intrinsic_target_reward: float):
        self.cfg = cfg
        self.intrinsic_target_reward = intrinsic_target_reward
        self.obstacles = None
        self.starts = None
        self.planner = None

    def add_grid_obstacles(self, obstacles, starts):
        self.obstacles = obstacles
        self.starts = starts
        self.planner = None

    def step(self, observations):
        if self.planner is None:
            cfg = self.cfg
            self.planner = multi_planner(self.obstacles, cfg.use_static_cost, cfg.use_dynamic_cost, cfg.reset_dynamic_cost,
                                         [tuple(s) for s in self.starts], self.intrinsic_target_reward)
            self.planner.set_crisscross(cfg.crisscross_ratio, cfg.crisscross_all)
            self._set_eval_options(cfg)
            if cfg.use_static_cost:
                obs_radius = len(observations[0]['obstacles']) // 2
                self.planner.set_penalties(np.asarray(get_penalties(self.obstacles, obs_radius, cfg), dtype=np.float32))

        agents = np.stack([o['agents'] for o in observations]).astype(np.float32, copy=False)
        obstacles = np.stack([o['obstacles'] for o in observations]).astype(np.float32, copy=False)
        xy = np.array([o['xy'] for o in observations], dtype=np.int64)
        target_xy = np.array([o['target_xy'] for o in observations], dtype=np.int64)

        rewards = self.planner.step(agents, obstacles, xy, target_xy)

        for k, obs in enumerate(observations):
            obs['agents'] = agents[k]
            obs['obstacles'] = obstacles[k]
        return rewards.tolist()

    def _set_eval_options(self, cfg):
        """Guidance table, density gate and dflow of the new multi_planner, as the C++ Follower's init does."""
        gate = cfg.guidance_gate_rho0 is not None or cfg.guidance_gate_rho1 is not None
        if cfg.guidance_dir is not None:
            from follower_cpp.guidance import load_table
            table = load_table(cfg.guidance_dir, self.obstacles, len(self.starts))
            self.planner.set_guidance(table.ravel().tolist())
        if gate:
            assert cfg.guidance_dir is not None, 'the density gate needs guidance_dir'
            assert cfg.guidance_gate_rho0 is not None and cfg.guidance_gate_rho1 is not None
            assert cfg.guidance_gate_rho1 > cfg.guidance_gate_rho0
            self.planner.set_guidance_gate(cfg.guidance_gate_rho0, cfg.guidance_gate_rho1, cfg.guidance_gate_halflife)
        if cfg.dflow != 'off':
            self.planner.set_dflow(DFLOW_MODES[cfg.dflow], cfg.dflow_with, cfg.dflow_opp)
        dump = (f'[planner config python] static={cfg.use_static_cost} dynamic={cfg.use_dynamic_cost} '
                f'reset={cfg.reset_dynamic_cost} crisscross_ratio={cfg.crisscross_ratio} '
                f'crisscross_all={cfg.crisscross_all} guidance_dir={cfg.guidance_dir} guidance_gate={gate} '
                f'gate_rho0={cfg.guidance_gate_rho0} gate_rho1={cfg.guidance_gate_rho1} '
                f'gate_halflife={cfg.guidance_gate_halflife} dflow={cfg.dflow} dflow_with={cfg.dflow_with} '
                f'dflow_opp={cfg.dflow_opp} fix_soft_positions={os.environ.get("LTF_FIX_SOFT_POSITIONS")}')
        if dump not in _CONFIGS_DUMPED:
            _CONFIGS_DUMPED.add(dump)
            print(dump, flush=True)


class ResettablePlanner:
    def __init__(self, cfg: PlannerConfig):
        self._cfg = cfg
        self._agent = None

    def update(self, observations):
        return self._agent.update(observations)

    def get_path(self):
        return self._agent.get_path()

    def reset_states(self, ):
        self._agent = Planner(self._cfg)
