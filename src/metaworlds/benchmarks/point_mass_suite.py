import gym
import numpy as np

from metaworlds.worlds.dm_control import PointMassWorld


def dict_template(d, **kwargs):
    e = d.copy()
    e.update(kwargs)
    return e


class AgentDynamics:
    mass_space = PointMassWorld.pomdp_space.spaces['mass']
    mass_train = gym.spaces.Box(mass_space.low, mass_space.high / 2)
    mass_test = mass_space

    def yield_training_set(self, n):
        for m in np.linspace(self.mass_train.low, self.mass_train.high, n):
            yield dict_template(PointMassWorld.default_pomdp, mass=m)

    def yield_test_set(self, n):
        for m in np.linspace(self.mass_test.low, self.mass_test.high, n):
            yield dict_template(PointMassWorld.default_pomdp, mass=m)


class EnvironmentDynamics:
    radius_space = PointMassWorld.pomdp_space.spaces['obstacle_radius']
    radius_train = gym.spaces.Box(radius_space.low, radius_space.high / 2)
    radius_test = radius_space

    def yield_training_set(self, n):
        for r in np.linspace(self.radius_train.low, self.radius_train.high, n):
            yield dict_template(
                PointMassWorld.default_pomdp, obstacle_radius=r)

    def yield_test_set(self, n):
        for r in np.linspace(self.radius_test.low, self.radius_test.high, n):
            yield dict_template(
                PointMassWorld.default_pomdp, obstacle_radius=r)


class RewardFunction:
    goal_space = PointMassWorld.pomdp_space.spaces['goal']
    radius = np.linalg.norm(goal_space.high) / 2

    def yield_training_set(self, n):
        for g in self._circle(self.radius, n):
            yield dict_template(PointMassWorld.default_pomdp, goal=g)

    def yield_test_set(self, n):
        offset = np.pi / n
        for g in self._circle(self.radius, n, offset):
            yield dict_template(PointMassWorld.default_pomdp, goal=g)

    @staticmethod
    def _circle(r, n, offset=0):
        for t in np.arange(0 + offset, 2 * np.pi + offset, 2 * np.pi / n):
            yield np.array([r * np.sin(t), r * np.cos(t)], dtype=np.float32)


class ObservationModel:
    noise_space = PointMassWorld.pomdp_space.spaces['observation_noise_std']
    noise_train = gym.spaces.Box(noise_space.low, noise_space.high / 2)
    noise_test = noise_space

    def yield_training_set(self, n):
        for s in np.linspace(self.noise_train.low, self.noise_train.high, n):
            yield dict_template(
                PointMassWorld.default_pomdp, observation_noise_std=s)

    def yield_test_set(self, n):
        for s in np.linspace(self.noise_test.low, self.noise_test.high, n):
            yield dict_template(
                PointMassWorld.default_pomdp, observation_noise_std=s)


class PointMassSuite:
    world = PointMassWorld
    agent_dynamics = AgentDynamics()
    environment_dynamics = EnvironmentDynamics()
    reward_function = RewardFunction()
    observation_model = ObservationModel()
    all_benchmarks = [
        agent_dynamics,
        environment_dynamics,
        reward_function,
        observation_model,
    ]
