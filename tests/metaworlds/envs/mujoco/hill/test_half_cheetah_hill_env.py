import pickle
import unittest

from metaworlds.envs.mujoco.hill.half_cheetah_hill_env import HalfCheetahHillEnv
from tests.helpers import step_env


class TestHalfCheetahHillEnv(unittest.TestCase):
    def test_pickleable(self):
        env = HalfCheetahHillEnv(regen_terrain=False)
        round_trip = pickle.loads(pickle.dumps(env))
        assert round_trip
        assert round_trip.difficulty == env.difficulty
        step_env(round_trip)
        round_trip.close()
        env.close()

    def test_does_not_modify_action(self):
        env = HalfCheetahHillEnv(regen_terrain=False)
        a = env.action_space.sample()
        a_copy = a.copy()
        env.reset()
        env.step(a)
        self.assertEqual(a.all(), a_copy.all())
        env.close()
