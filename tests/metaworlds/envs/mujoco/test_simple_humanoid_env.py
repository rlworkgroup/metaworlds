import pickle
import unittest

from metaworlds.envs.mujoco.simple_humanoid_env import SimpleHumanoidEnv
from tests.helpers import step_env


class TestSimpleHumanoidEnv(unittest.TestCase):
    def test_pickleable(self):
        env = SimpleHumanoidEnv(alive_bonus=1.0)
        round_trip = pickle.loads(pickle.dumps(env))
        assert round_trip
        assert round_trip.alive_bonus == env.alive_bonus
        step_env(round_trip)
        round_trip.close()
        env.close()

    def test_does_not_modify_action(self):
        env = SimpleHumanoidEnv(alive_bonus=1.0)
        a = env.action_space.sample()
        a_copy = a.copy()
        env.reset()
        env.step(a)
        self.assertEqual(a.all(), a_copy.all())
        env.close()
