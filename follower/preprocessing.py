import numpy as np
import gymnasium
from gymnasium import ObservationWrapper
from gymnasium.spaces import Box, Dict

from follower.planning import BatchPlanner, PlannerConfig


class PreprocessorConfig(PlannerConfig):
    network_input_radius: int = 5
    intrinsic_target_reward: float = 0.01
    # add the agent's short-term memory of its own motion as two input channels, see MemoryWrapper
    use_memory: bool = False


# number of past steps the memory channels cover; follower_cpp/follower.cpp uses the same value
MEMORY_STEPS = 8


def follower_preprocessor(env, algo_config):
    env = wrap_preprocessors(env, algo_config.training_config.preprocessing)
    return env


def wrap_preprocessors(env, config: PreprocessorConfig, auto_reset=False):
    env = FollowerWrapper(env=env, config=config)
    if config.use_memory:
        env = MemoryWrapper(env)
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
        return '2_' + x


class AutoResetWrapper(gymnasium.Wrapper):
    def step(self, action):
        observations, rewards, terminated, truncated, infos = self.env.step(action)
        if all(terminated) or all(truncated):
            observations, _ = self.env.reset()
        return observations, rewards, terminated, truncated, infos
