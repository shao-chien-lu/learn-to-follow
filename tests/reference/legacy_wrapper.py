"""Frozen copy of FollowerWrapper from git commit 1b50c12 (python per-agent loop), the oracle for the C++ version."""
from gymnasium import ObservationWrapper

from follower.planning import ResettablePlanner


class LegacyFollowerWrapper(ObservationWrapper):

    def __init__(self, env, config):
        super().__init__(env)
        self._cfg = config
        self.re_plan = ResettablePlanner(self._cfg)
        self.prev_goals = None
        self.intrinsic_reward = None

    @staticmethod
    def get_relative_xy(x, y, tx, ty, obs_radius):
        dx, dy = x - tx, y - ty
        if dx > obs_radius or dx < -obs_radius or dy > obs_radius or dy < -obs_radius:
            return None, None
        return obs_radius - dx, obs_radius - dy

    def observation(self, observations):
        self.re_plan.update(observations)
        paths = self.re_plan.get_path()

        new_goals = []
        intrinsic_rewards = []

        for k, path in enumerate(paths):
            obs = observations[k]

            if path is None:
                new_goals.append(obs['target_xy'])
                path = []
            else:
                subgoal_achieved = self.prev_goals and obs['xy'] == self.prev_goals[k]
                intrinsic_rewards.append(self._cfg.intrinsic_target_reward if subgoal_achieved else 0.0)
                new_goals.append(path[1])

            obs['obstacles'][obs['obstacles'] > 0] *= -1

            r = obs['obstacles'].shape[0] // 2
            for idx, (gx, gy) in enumerate(path):
                x, y = self.get_relative_xy(*obs['xy'], gx, gy, r)
                if x is not None and y is not None:
                    obs['obstacles'][x, y] = 1.0
                else:
                    break
        self.prev_goals = new_goals
        self.intrinsic_reward = intrinsic_rewards

        return observations

    def get_intrinsic_rewards(self, reward):
        for agent_idx, r in enumerate(reward):
            reward[agent_idx] = self.intrinsic_reward[agent_idx]
        return reward

    def step(self, action):
        observation, reward, done, tr, info = self.env.step(action)
        return self.observation(observation), self.get_intrinsic_rewards(reward), done, tr, info

    def reset_state(self):
        self.re_plan.reset_states()
        self.re_plan._agent.add_grid_obstacles(self.get_global_obstacles(), self.get_global_agents_xy())

        self.prev_goals = None
        self.intrinsic_reward = None

    def reset(self, **kwargs):
        observations, infos = self.env.reset(**kwargs)
        self.reset_state()
        return self.observation(observations), infos
