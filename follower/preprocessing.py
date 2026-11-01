from collections import deque

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

import numpy as np
import gymnasium
from gymnasium import ObservationWrapper
from gymnasium.spaces import Box, Dict

import follower.pogema_fix  # noqa: F401  (patches pogema when LTF_FIX_SOFT_POSITIONS=1)
from follower.planning import BatchPlanner, PlannerConfig


class PreprocessorConfig(PlannerConfig):
    network_input_radius: int = 5
    intrinsic_target_reward: float = 0.01
    # add the agent's short-term memory of its own motion as two input channels, see MemoryWrapper
    use_memory: bool = False
    # add what the agent saw of the other agents on the last steps as input channels, see NeighbourMemoryWrapper:
    # 'trail' (2 channels: their recent cells and how long each one has stayed)
    neighbour_memory: Literal['none', 'trail'] = 'none'


# number of past steps the memory channels cover; follower_cpp/follower.cpp uses the same value
MEMORY_STEPS = 8


def follower_preprocessor(env, algo_config):
    env = wrap_preprocessors(env, algo_config.training_config.preprocessing)
    return env


def wrap_preprocessors(env, config: PreprocessorConfig, auto_reset=False):
    env = FollowerWrapper(env=env, config=config)
    if config.use_memory:
        env = MemoryWrapper(env)
    if config.neighbour_memory != 'none':
        env = NeighbourMemoryWrapper(env)
    env = CutObservationWrapper(env, target_observation_radius=config.network_input_radius)
    env = ConcatPositionalFeatures(env)
    if auto_reset:
        env = AutoResetWrapper(env)
    return env


class FollowerWrapper(ObservationWrapper):

    def __init__(self, env, config: PreprocessorConfig):
        super().__init__(env)
        self._cfg: PreprocessorConfig = config
        self.planner = BatchPlanner(self._cfg, self._cfg.intrinsic_target_reward)
        self.intrinsic_reward = None

    def observation(self, observations):
        # One native call re-plans every agent (A* with occupation-based costs), sets obstacles to -1.0, draws each
        # agent's path as +1.0 into its obstacle channel, and returns the intrinsic rewards (sub-goal reached).
        self.intrinsic_reward = self.planner.step(observations)
        return observations

    def get_intrinsic_rewards(self, reward):
        for agent_idx, r in enumerate(reward):
            reward[agent_idx] = self.intrinsic_reward[agent_idx]
        return reward

    def step(self, action):
        observation, reward, done, tr, info = self.env.step(action)
        return self.observation(observation), self.get_intrinsic_rewards(reward), done, tr, info

    def reset_state(self):
        self.planner = BatchPlanner(self._cfg, self._cfg.intrinsic_target_reward)
        self.planner.add_grid_obstacles(self.get_global_obstacles(), self.get_global_agents_xy())

        self.intrinsic_reward = None

    def reset(self, **kwargs):
        observations, infos = self.env.reset(**kwargs)
        self.reset_state()
        return self.observation(observations), infos


class MemoryWrapper(gymnasium.Wrapper):
    """
    Short-term memory of each agent's own motion, as two extra channels in the full observation window (cropped and
    stacked after 'obstacles' and 'agents' like them):
      'blocked': if the agent did not move on the last step, the cell its last action aimed at (its own cell for
                 'stay'), valued min(steps without moving, MEMORY_STEPS) / MEMORY_STEPS;
      'trail':   its positions of the last MEMORY_STEPS steps, valued (MEMORY_STEPS + 1 - age) / MEMORY_STEPS, the most
                 recent one where they coincide.
    follower_cpp/follower.cpp builds the same channels at inference.
    """

    def __init__(self, env):
        super().__init__(env)
        d = self.env.observation_space['obstacles'].shape[0]
        self.observation_space = Dict({**self.env.observation_space.spaces,
                                       'blocked': Box(0.0, 1.0, shape=(d, d)), 'trail': Box(0.0, 1.0, shape=(d, d))})
        self._moves = self.env.grid_config.MOVES
        self._last_xy, self._last_actions, self._trails, self._stuck = None, None, None, None

    def reset(self, **kwargs):
        observations, infos = self.env.reset(**kwargs)
        self._last_xy = [tuple(o['xy']) for o in observations]
        self._last_actions = [None] * len(observations)
        self._trails = [[] for _ in observations]  # previous positions, most recent first
        self._stuck = [0] * len(observations)
        return self._add_channels(observations), infos

    def step(self, action):
        self._last_actions = list(action)  # copy: pogema's soft collisions overwrite blocked moves with 'stay' in place
        observations, reward, terminated, truncated, infos = self.env.step(action)
        for k, obs in enumerate(observations):
            xy = tuple(obs['xy'])
            self._stuck[k] = self._stuck[k] + 1 if xy == self._last_xy[k] else 0
            self._trails[k] = [self._last_xy[k]] + self._trails[k][:MEMORY_STEPS - 1]
            self._last_xy[k] = xy
        return self._add_channels(observations), reward, terminated, truncated, infos

    def _add_channels(self, observations):
        r = self.observation_space['blocked'].shape[0] // 2
        for k, obs in enumerate(observations):
            x, y = self._last_xy[k]
            blocked = np.zeros((2 * r + 1, 2 * r + 1), dtype=np.float32)
            if self._stuck[k]:
                dx, dy = self._moves[self._last_actions[k]]
                blocked[r + dx, r + dy] = min(self._stuck[k], MEMORY_STEPS) / MEMORY_STEPS
            trail = np.zeros_like(blocked)
            for age in range(len(self._trails[k]), 0, -1):  # oldest first, so the most recent one wins
                px, py = self._trails[k][age - 1]
                if abs(px - x) <= r and abs(py - y) <= r:
                    trail[r + px - x, r + py - y] = (MEMORY_STEPS + 1 - age) / MEMORY_STEPS
            obs['blocked'], obs['trail'] = blocked, trail
        return observations


class NeighbourMemoryWrapper(gymnasium.Wrapper):
    """
    Memory of the other agents, from the agent's own past 'agents' channels only (each one as seen from where the agent
    was then, without its own cell), moved into the current window. A cell counts as seen occupied a steps ago only if
    it was in the agent's window then. Two channels, in the full observation window (cropped and stacked last):
      'nb_trail': cells seen occupied on the last MEMORY_STEPS steps, valued (MEMORY_STEPS + 1 - age) / MEMORY_STEPS,
                  the most recent one where they coincide;
      'nb_wait':  for each currently occupied cell, the number of directly preceding steps it was seen occupied,
                  / MEMORY_STEPS (0: someone just arrived there, 1: someone has stayed).
    follower_cpp/follower.cpp builds the same channels at inference.
    """

    def __init__(self, env):
        super().__init__(env)
        d = self.env.observation_space['obstacles'].shape[0]
        self.observation_space = Dict({**self.env.observation_space.spaces,
                                       'nb_trail': Box(0.0, 1.0, shape=(d, d)), 'nb_wait': Box(0.0, 1.0, shape=(d, d))})
        self._history = None  # (xy, seen) of the previous steps, most recent first
        self._current = None

    def reset(self, **kwargs):
        observations, infos = self.env.reset(**kwargs)
        self._history = deque(maxlen=MEMORY_STEPS)
        self._current = self._snapshot(observations)
        return self._add_channels(observations), infos

    def step(self, action):
        observations, reward, terminated, truncated, infos = self.env.step(action)
        self._history.appendleft(self._current)
        self._current = self._snapshot(observations)
        return self._add_channels(observations), reward, terminated, truncated, infos

    @staticmethod
    def _snapshot(observations):
        # xy is relative to each agent's own start, which is all that is needed to move its own past windows
        xy = np.array([o['xy'] for o in observations], dtype=np.int64)
        seen = np.array([o['agents'] for o in observations]) > 0
        r = seen.shape[1] // 2
        seen[:, r, r] = False
        return xy, seen

    def _add_channels(self, observations):
        xy, now = self._current
        n, d = now.shape[0], now.shape[1]
        idx = np.arange(d)
        agent = np.arange(n)[:, None, None]
        past = []  # per age 1, 2, ...: (n, d, d) cells of the current window seen occupied then
        for prev_xy, prev_seen in self._history:
            # current cell i (offset i - r) is cell i + xy - prev_xy of the window back then
            si = idx[None, :, None] + (xy[:, 0] - prev_xy[:, 0])[:, None, None]
            sj = idx[None, None, :] + (xy[:, 1] - prev_xy[:, 1])[:, None, None]
            inside = (si >= 0) & (si < d) & (sj >= 0) & (sj < d)
            past.append(prev_seen[agent, si.clip(0, d - 1), sj.clip(0, d - 1)] & inside)
        trail = np.zeros((n, d, d), dtype=np.float32)
        for age in range(len(past), 0, -1):  # oldest first, so the most recent one wins
            trail[past[age - 1]] = (MEMORY_STEPS + 1 - age) / MEMORY_STEPS
        staying, steps = now.copy(), np.zeros((n, d, d), dtype=np.float32)
        for seen in past:
            staying &= seen
            steps += staying
        for k, obs in enumerate(observations):
            obs['nb_trail'], obs['nb_wait'] = trail[k], steps[k] / MEMORY_STEPS
        return observations


class CutObservationWrapper(ObservationWrapper):
    def __init__(self, env, target_observation_radius):
        super().__init__(env)
        self._target_obs_radius = target_observation_radius
        self._initial_obs_radius = self.env.observation_space['obstacles'].shape[0] // 2

        for key, value in self.observation_space.items():
            d = self._initial_obs_radius * 2 + 1
            if value.shape == (d, d):
                r = self._target_obs_radius
                self.observation_space[key] = Box(0.0, 1.0, shape=(r * 2 + 1, r * 2 + 1))

    def observation(self, observations):
        tr = self._target_obs_radius
        ir = self._initial_obs_radius
        d = ir * 2 + 1

        for obs in observations:
            for key, value in obs.items():
                if hasattr(value, 'shape') and value.shape == (d, d):
                    obs[key] = value[ir - tr:ir + tr + 1, ir - tr:ir + tr + 1]

        return observations


class ConcatPositionalFeatures(ObservationWrapper):

    def __init__(self, env):
        super().__init__(env)
        self.to_concat = []

        observation_space = Dict()
        full_size = self.env.observation_space['obstacles'].shape[0]

        for key, value in self.observation_space.items():
            if value.shape == (full_size, full_size):
                self.to_concat.append(key)
            else:
                observation_space[key] = value

        obs_shape = (len(self.to_concat), full_size, full_size)
        observation_space['obs'] = Box(0.0, 1.0, shape=obs_shape)
        self.to_concat.sort(key=self.key_comparator)
        self.observation_space = observation_space

    def observation(self, observations):
        for agent_idx, obs in enumerate(observations):
            main_obs = np.concatenate([obs[key][None] for key in self.to_concat])
            for key in self.to_concat:
                del obs[key]

            for key in obs:
                obs[key] = np.array(obs[key], dtype=np.float32)
            observations[agent_idx]['obs'] = main_obs.astype(np.float32)
        return observations

    @staticmethod
    def key_comparator(x):
        if x == 'obstacles':
            return '0_' + x
        elif 'agents' in x:
            return '1_' + x
        elif x.startswith('nb_'):  # NeighbourMemoryWrapper's channels go after the own memory
            return '3_' + x
        return '2_' + x


class AutoResetWrapper(gymnasium.Wrapper):
    def step(self, action):
        observations, rewards, terminated, truncated, infos = self.env.step(action)
        if all(terminated) or all(truncated):
            observations, _ = self.env.reset()
        return observations, rewards, terminated, truncated, infos
