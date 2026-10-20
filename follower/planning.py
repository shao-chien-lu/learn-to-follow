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
