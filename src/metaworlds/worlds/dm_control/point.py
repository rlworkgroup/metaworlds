from dm_control import mjcf
from dm_control.rl import control
from dm_control.suite import point_mass
from dm_control.utils import rewards
from garage.envs.dm_control import DmControlEnv
import gym.spaces
import numpy as np

from metaworlds.worlds.core import ParametricWorld


class PointMassWorld(ParametricWorld, DmControlEnv):

    pomdp_space = gym.spaces.Dict({
        'goal':
        gym.spaces.Box(-0.25, 0.25, (2, ), dtype=np.float32),
        'mass':
        gym.spaces.Box(0.1, 10.0, (), dtype=np.float32),
        'observation_noise_std':
        gym.spaces.Box(0., 0.25, (), dtype=np.float32),
        'obstacle_radius':
        gym.spaces.Box(0.0, 0.1, (), dtype=np.float32),
    })

    default_pomdp = {
        'goal': np.array((0.25, 0.25), dtype=np.float32),
        'mass': np.array(0.3, dtype=np.float32),
        'observation_noise_std': np.array(0., dtype=np.float32),
        'obstacle_radius': np.array(0., dtype=np.float32),
    }

    def __init__(self):
        self._pomdp = self.default_pomdp
        self._task = PointMassTask(False)
        physics = PointMassPhysics.from_pomdp(self._pomdp)
        env = control.Environment(
            physics, self._task, time_limit=point_mass._DEFAULT_TIME_LIMIT)
        DmControlEnv.__init__(self, env)

    @property
    def pomdp(self):
        return self._pomdp

    @pomdp.setter
    def pomdp(self, pomdp_descriptor):
        super()._validate_descriptor(pomdp_descriptor)
        old_pomdp = self._pomdp
        self._pomdp = pomdp_descriptor
        self._task.observation_noise_std = self._pomdp['observation_noise_std']
        if self._physics_changed(old_pomdp, self._pomdp):
            new_physics = PointMassPhysics.from_pomdp(pomdp_descriptor)
            self._env = control.Environment(
                new_physics,
                self._task,
                time_limit=point_mass._DEFAULT_TIME_LIMIT)

            if self._viewer:
                self._viewer.close()
                self._viewer = None

    @staticmethod
    def _physics_changed(old, new):
        return not (np.array_equal(old['goal'], new['goal'])
                    and old['mass'] == new['mass']
                    and old['obstacle_radius'] == new['obstacle_radius'])


class PointMassPhysics(point_mass.Physics, mjcf.Physics):
    @classmethod
    def get_mjcf(cls):
        xml_string, assets = point_mass.get_model_and_assets()
        m = mjcf.from_xml_string(xml_string, assets=assets)

        # Turn collision on
        m.option.flag.contact = 'enable'

        # Use direct motors to contol the point, not tendons
        m.actuator.motor['t1'].tendon = None
        m.actuator.motor['t1'].joint = m.worldbody.body['pointmass'].joint[
            'root_x']
        m.actuator.motor['t2'].tendon = None
        m.actuator.motor['t2'].joint = m.worldbody.body['pointmass'].joint[
            'root_y']

        return m

    @classmethod
    def from_pomdp(cls, pomdp_descriptor):
        m = cls.get_mjcf()

        m.worldbody.body['pointmass'].geom[
            'pointmass'].mass = pomdp_descriptor['mass']
        m.worldbody.geom['target'].pos[:2] = pomdp_descriptor['goal']
        if pomdp_descriptor['obstacle_radius'] != 0:
            x, y, r = 0.125, 0.125, pomdp_descriptor['obstacle_radius']
            m.worldbody.add(
                'geom',
                name='obstacle',
                type='cylinder',
                size='{} {}'.format(float(r), 0.1),
                pos='{} {} {}'.format(float(x), float(y), 0.0),
                material='decoration')

        return cls.from_mjcf_model(m)


class PointMassTask(point_mass.PointMass):
    def __init__(self, randomize_gains, random=None):
        super().__init__(randomize_gains, random=random)
        self._observation_noise_std = 0.

    def initialize_episode(self, physics):
        initial_state = np.zeros_like(physics.state())
        physics.set_state(initial_state)

    def get_reward(self, physics):
        """Replace dm_control's shaped reward with a simple L2 norm cost"""
        target_size = physics.named.model.geom_size['target', 0]
        return -physics.mass_to_target_dist()

    def get_observation(self, physics):
        obs = super().get_observation(physics)
        for k, v in obs.items():
            obs[k] = obs[k] + np.random.normal(
                0, self._observation_noise_std, size=obs[k].shape)
        return obs

    @property
    def observation_noise_std(self):
        return self._observation_noise_std

    @observation_noise_std.setter
    def observation_noise_std(self, std):
        self._observation_noise_std = std
