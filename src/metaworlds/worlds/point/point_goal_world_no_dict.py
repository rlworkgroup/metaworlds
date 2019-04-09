import gym
import gym.spaces
import numpy as np

from world import GoalConditionedWorld
import util


class PointGoalWorld(GoalConditionedWorld):
    """A goal-conditioned point-mass world."""

    observation_space = gym.spaces.Box(
        low=-10.0, high=10.0, shape=(2, ), dtype=np.float32)
    action_space = gym.spaces.Box(
        low=-0.1, high=0.1, shape=(2, ), dtype=np.float32)
    goal_tolerance = action_space.low
    default_pomdp = {
        'start': np.array((0., 0.), dtype=np.float32),
        'goal': np.array((1., 1.), dtype=np.float32),
    }

    def __init__(self):
        super().__init__(self.observation_space, self.observation_space)
        self.pomdp = self.default_pomdp
        self._point = np.copy(self._pomdp['start'])

    def reset(self):
        self._point = np.copy(self._pomdp['start'])
        return np.copy(self._point)

    def step(self, action):
        self._point = self._point + action

        dist = np.linalg.norm(self._point - self._pomdp['goal'])
        reward = -dist
        done = dist < np.linalg.norm(self.goal_tolerance)

        return np.copy(self._point), reward, done, dict()

    def render(self, mode='human'):
        print(self._point)


if __name__ == '__main__':
    """Generate a series of random point environments and step them."""
    import pickle

    # Create an instance of the world
    world = PointGoalWorld()
    util.print_running(world)

    for _ in range(5):
        # Sample a new POMDP
        mdp = world.pomdp_space.sample()
        world.pomdp = mdp
        util.print_pomdp(mdp)

        # Reset and step the POMDP in the World
        initial_state = world.reset()
        util.print_reset(initial_state)
        for _ in range(10):
            step = world.step(world.action_space.low)
            util.print_step(step)
