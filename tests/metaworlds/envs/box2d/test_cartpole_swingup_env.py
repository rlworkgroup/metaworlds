import pickle
import unittest

from metaworlds.envs.box2d.cartpole_swingup_env import CartpoleSwingupEnv
from tests.helpers import step_env


class TestCartpoleSwingupEnv(unittest.TestCase):
    def test_pickleable(self):
        env = CartpoleSwingupEnv(obs_noise=1.0)
        round_trip = pickle.loads(pickle.dumps(env))
        assert round_trip
        assert round_trip.obs_noise == env.obs_noise
        step_env(round_trip)
        round_trip.close()
        env.close()

    def test_does_not_modify_action(self):
        env = CartpoleSwingupEnv(obs_noise=1.0)
        a = env.action_space.sample()
        a_copy = a.copy()
        env.reset()
        env.step(a)
        self.assertEqual(a.all(), a_copy.all())
        env.close()
