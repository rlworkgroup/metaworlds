import gym
import gym.spaces
import numpy as np

import util


class PointEnv(gym.Env):
    """A simple point mass environment."""
    observation_space = gym.spaces.Box(
        low=-np.inf, high=np.inf, shape=(2, ), dtype=np.float32)

    action_space = gym.spaces.Box(
        low=-0.1, high=0.1, shape=(2, ), dtype=np.float32)

    start = np.array((0., 0.), dtype=np.float32)
    goal = np.array((1., 1.), dtype=np.float32)
    goal_tolerance = np.linalg.norm(action_space.low)

    def __init__(self):
        self._point = np.copy(self.start)

    def reset(self):
        self._point = np.copy(self.start)
        return np.copy(self._point)

    def step(self, action):
        self._point = self._point + action

        dist = np.linalg.norm(self._point - self.goal)
        reward = -dist
        done = dist < np.linalg.norm(self.goal_tolerance)

        return np.copy(self._point), reward, done, dict()

    def render(self, mode='human'):
        print(self._point)


if __name__ == '__main__':
    """Constructs and steps the Point environment."""
    # Create environment
    env = PointEnv()
    util.print_running(env)

    # Reset and step environment
    for _ in range(5):
        initial_state = env.reset()
        util.print_reset(initial_state)

        for _ in range(10):
            step = env.step(env.action_space.high)
            util.print_step(step)