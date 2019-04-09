import gym
import gym.spaces
import numpy as np

from world import MultiTaskWorld
import util
from util.reward_functions import sparse_goal, dense_goal, mostly_dense_goal

_START = start = np.array((0., 0.), dtype=np.float32)
_GOAL = np.array((1., 1.), dtype=np.float32)
_GOAL_TOLERANCE = 0.1
_DEFAULT_POMDP = 0


class MultiTaskPointWorld(MultiTaskWorld):
    """A general multi-task point-mass world."""

    action_space = gym.spaces.Box(
        low=-0.1, high=0.1, shape=(2, ), dtype=np.float32)
    observation_space = gym.spaces.Box(
        low=-np.inf, high=np.inf, shape=(2, ), dtype=np.float32)

    def __init__(self):
        super().__init__(
            sparse_goal(_GOAL, _GOAL_TOLERANCE), dense_goal(_GOAL),
            mostly_dense_goal(_GOAL, _GOAL_TOLERANCE))
        self._point = np.copy(_START)
        self.pomdp = _DEFAULT_POMDP

    def reset(self):
        self._point = np.copy(_START)
        return np.copy(self._point)

    def step(self, action):
        point_old = np.copy(self._point)
        self._point = self._point + action

        reward = self._reward(point_old, action, self._point)
        dist = np.linalg.norm(self._point - _GOAL)
        done = dist < _GOAL_TOLERANCE

        return np.copy(self._point), reward, done, dict()

    def render(self, mode='human'):
        print(self._point)


if __name__ == '__main__':
    """Generate a series of random point environments and step them."""
    import pickle

    # Create an instance of the world
    world = MultiTaskPointWorld()
    util.print_running(world)

    for _ in range(5):
        # Sample a new POMDP
        mdp = world.pomdp_space.sample()
        mdp = pickle.loads(pickle.dumps(mdp))  # the POMDP is pickleable!
        world.pomdp = mdp
        util.print_pomdp(mdp)

        # Reset and step the POMDP in the World
        initial_state = world.reset()
        util.print_reset(initial_state)
        for _ in range(10):
            step = world.step(world.action_space.high)
            util.print_step(step)
