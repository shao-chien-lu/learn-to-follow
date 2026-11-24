from follower.inference import FollowerInferenceConfig
from follower.preprocessing import PreprocessorConfig
# noinspection PyUnresolvedReferences
from utils import fix_num_threads_issue

import multiprocessing
import os
from pathlib import Path
from typing import Optional

from sample_factory.utils.utils import log

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

from pydantic import Extra


class FollowerConfigCPP(FollowerInferenceConfig, extra=Extra.forbid):
    name: Literal['FollowerLite'] = 'FollowerLite'
    num_process: int = 8
    num_threads: int = 8
    path_to_weights: str = "model/follower-lite/"
    preprocessing: PreprocessorConfig = PreprocessorConfig()
    # eval-time crisscross highways of the planner (None: the model's training setting)
    crisscross_ratio: Optional[float] = None
    crisscross_all: Optional[bool] = None
    # eval-time per-map guidance tables (follower_cpp/guidance.py), replacing the crisscross factor: a directory of
    # <guidance_key(map, num_agents)>.npy; every evaluated map and agent count must have one (None: off)
    guidance_dir: Optional[str] = None
    # density gate of the guidance tables (None: off, the tables apply fully): every agent blends crisscross and table
    # cost factors in log space with g = clip((rho - rho0) / (rho1 - rho0), 0, 1), rho = the other agents in its own
    # observation window / its free cells, EMA-smoothed with half-life guidance_gate_halflife steps (<= 0: raw rho)
    guidance_gate_rho0: Optional[float] = None
    guidance_gate_rho1: Optional[float] = None
    guidance_gate_halflife: float = 4.0
    # eval-time direction-aware dynamic cost from the observed flow of other agents (planner::set_dflow): 'off' keeps
    # the planner unchanged; 'split' weighs observations moving with / against a step by dflow_with / dflow_opp
    dflow: Literal['off', 'add', 'split', 'ggboost'] = 'off'
    dflow_with: float = 1.0
    dflow_opp: float = 1.0


# one line per distinct planner config and process, as evidence of the settings that reached the C++ side
_CONFIGS_DUMPED = set()


class FollowerInferenceCPP:
    def __init__(self, cfg: FollowerConfigCPP):
        self.cfg: FollowerConfigCPP = cfg

        # noinspection PyUnresolvedReferences
        import cppimport.import_hook
        # noinspection PyUnresolvedReferences
        from follower_cpp.follower import Follower
        # noinspection PyUnresolvedReferences
        from follower_cpp.config import Config

        self.algo = Follower()
        self.cpp_config = Config()
        self.cpp_config.obs_radius = self.cfg.training_config.environment.grid_config.obs_radius
        if self.cfg.num_threads > multiprocessing.cpu_count():
            log.warning(f'Setting num_threads to {multiprocessing.cpu_count()}, based on CPU count.')
            self.cfg.num_threads = multiprocessing.cpu_count()
        self.cpp_config.num_threads = self.cfg.num_threads
        self.cpp_config.use_static_cost = self.cfg.training_config.preprocessing.use_static_cost
        self.cpp_config.use_dynamic_cost = self.cfg.training_config.preprocessing.use_dynamic_cost
        self.cpp_config.reset_dynamic_cost = self.cfg.training_config.preprocessing.reset_dynamic_cost
        for key in ['crisscross_ratio', 'crisscross_all']:
            value = getattr(self.cfg, key)
            setattr(self.cpp_config, key, getattr(self.cfg.training_config.preprocessing, key) if value is None else value)
        self.cpp_config.seed = self.cfg.seed
        self.cpp_config.use_memory = self.cfg.training_config.preprocessing.use_memory
        self.cpp_config.neighbour_memory = self.cfg.training_config.preprocessing.neighbour_memory
        self.cpp_config.ablate_channels = self.cfg.training_config.preprocessing.ablate_channels
        if self.cfg.guidance_gate_rho0 is not None or self.cfg.guidance_gate_rho1 is not None:
            assert self.cfg.guidance_dir is not None, 'the density gate needs guidance_dir'
            assert self.cfg.guidance_gate_rho0 is not None and self.cfg.guidance_gate_rho1 is not None
            assert self.cfg.guidance_gate_rho1 > self.cfg.guidance_gate_rho0
            self.cpp_config.guidance_gate = True
            self.cpp_config.gate_rho0 = self.cfg.guidance_gate_rho0
            self.cpp_config.gate_rho1 = self.cfg.guidance_gate_rho1
            self.cpp_config.gate_halflife = self.cfg.guidance_gate_halflife
        self.cpp_config.dflow = self.cfg.dflow
        self.cpp_config.dflow_with = self.cfg.dflow_with
        self.cpp_config.dflow_opp = self.cfg.dflow_opp

        w_dir = Path(self.cfg.path_to_weights)
        self.cpp_config.path_to_weights = str(w_dir / str(w_dir.name + '.onnx'))

        c = self.cpp_config
        dump = (f'[planner config] weights={c.path_to_weights} static={c.use_static_cost} dynamic={c.use_dynamic_cost} '
                f'reset={c.reset_dynamic_cost} crisscross_ratio={c.crisscross_ratio} crisscross_all={c.crisscross_all} '
                f'guidance_dir={self.cfg.guidance_dir} guidance_gate={c.guidance_gate} gate_rho0={c.gate_rho0} '
                f'gate_rho1={c.gate_rho1} gate_halflife={c.gate_halflife} dflow={c.dflow} dflow_with={c.dflow_with} '
                f'dflow_opp={c.dflow_opp} fix_soft_positions={os.environ.get("LTF_FIX_SOFT_POSITIONS")}')
        if dump not in _CONFIGS_DUMPED:
            _CONFIGS_DUMPED.add(dump)
            print(dump, flush=True)

    def full_act(self, episode_length):
        return self.algo.full_act(episode_length)

    def act(self, observations):
        if 'global_obstacles' in observations[0]:
            agents_xy = [obs['global_agent_xy'] for obs in observations]
            if self.cfg.guidance_dir is not None:
                from follower_cpp.guidance import load_table
                table = load_table(self.cfg.guidance_dir, observations[0]['global_obstacles'], len(observations))
                self.cpp_config.guidance_table = table.ravel().tolist()
            self.algo.init(self.cpp_config, observations[0]['global_obstacles'], agents_xy)

        xy = [o['xy'] for o in observations]
        target_xy = [o['target_xy'] for o in observations]
        actions = self.algo.act(xy, target_xy)
        return actions

    def reset_states(self):
        pass
