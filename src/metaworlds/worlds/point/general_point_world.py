import gym
import gym.spaces
import numpy as np

from world import GeneralizedWorld
import util
from util.observation_functions import identity, gaussian_noise, gumbel_noise
from util.reward_functions import sparse_goal, dense_goal, mostly_dense_goal
from util.termination_functions import never, goal_state, noisy_goal_state
from util.transition_functions import deterministic, epsilon_random, walls

_EPSILON = 0.1
_BOUNDS = 5.0
_GOAL = np.array((1., 1.), dtype=np.float32)
_GOAL_TOLERANCE = 0.1

_reward_functions = [
    sparse_goal(_GOAL, _GOAL_TOLERANCE),
    dense_goal(_GOAL),
    mostly_dense_goal(_GOAL, _GOAL_TOLERANCE),
]

_observation_functions = [
    identity,
    gaussian_noise(_EPSILON),
    gumbel_noise(_EPSILON),
]

_transition_functions = [
    deterministic,
    epsilon_random(_EPSILON),
    walls([-_BOUNDS, _BOUNDS])
]

_termination_functions = [
    never,
    goal_state(_GOAL, _GOAL_TOLERANCE),
    noisy_goal_state(_GOAL, _GOAL_TOLERANCE, _EPSILON)
]


class GeneralPointWorld(GeneralizedWorld):
    """A completely-generalized point-mass world whose POMDP encodes the
    observation, transition, reward, and termination functions."""

    action_space = gym.spaces.Box(
        low=-0.1, high=0.1, shape=(2, ), dtype=np.float32)

    observation_space = gym.spaces.Box(
        low=-np.inf, high=np.inf, shape=(2, ), dtype=np.float32)

    start = np.array((0., 0.), dtype=np.float32)
    default_pomdp = dict(
        observation_function=0,
        transition_function=0,
        reward_function=0,
        termination_function=0)

    def __init__(self):
        super().__init__(_observation_functions, _transition_functions,
                         _reward_functions, _termination_functions)
        self._point = np.copy(self.start)
        self.pomdp = self.default_pomdp

    def reset(self):
        self._point = np.copy(self.start)
        return np.copy(self._point)

    def step(self, action):
        point_old = np.copy(self._point)

        self._point = self._transition(self._point, action)
        reward = self._reward(point_old, action, self._point)
        done = self._termination(self._point)

        return np.copy(self._point), reward, done, dict()

    def render(self, mode='human'):
        print(self._point)


if __name__ == '__main__':
    """Generate a series of random point environments and step them."""
    import pickle
    import time

    # Create an instance of the world
    world = GeneralPointWorld()
    util.print_running(world)

    for _ in range(5):
        # Sample a new POMDP
        mdp = world.pomdp_space.sample()
        mdp = pickle.loads(pickle.dumps(mdp))  # This POMDP is still pickleable
        world.pomdp = mdp
        util.print_pomdp(mdp)

        # Reset and step the POMDP in the world
        initial_state = world.reset()
        util.print_reset(initial_state)
        for _ in range(10):
            step = world.step(world.action_space.sample())
            util.print_step(step)
