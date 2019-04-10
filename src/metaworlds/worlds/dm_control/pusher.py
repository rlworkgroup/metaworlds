import collections

from dm_control import mjcf
from dm_control.rl import control
from dm_control.suite import base
from dm_control.utils import rewards
from garage.envs.dm_control import DmControlEnv
import gym.spaces
import numpy as np

from metaworlds import pallet
from metaworlds.worlds.core import ParametricWorld

_DEFAULT_TIME_LIMIT = 100


def get_model_and_assets():
    """Returns a tuple containing the model XML string and a dict of assets."""
    return pallet.read_model('pusher/pusher.xml'), pallet.ASSETS


class PusherWorld(ParametricWorld, DmControlEnv):
    pomdp_space = gym.spaces.Dict({
        'goal':
        gym.spaces.Box(-1., 1., (2, ), dtype=np.float32),
        'gear':
        gym.spaces.Box(15, 150., (), dtype=np.float32),
        'obstacle_radius':
        gym.spaces.Box(0.0, 0.25, (), dtype=np.float32),
        'observation_noise_std':
        gym.spaces.Box(0., 0.25, (), dtype=np.float32),
    })

    default_pomdp = {
        'goal': np.array((0., -1.0), dtype=np.float32),
        'gear': np.array(25., dtype=np.float32),
        'obstacle_radius': np.array(0., dtype=np.float32),
        'observation_noise_std': np.array(0., dtype=np.float32),
    }

    def __init__(self):
        self._pomdp = self.default_pomdp
        self._task = PusherTask(False)
        physics = PusherPhysics.from_pomdp(self._pomdp)
        env = control.Environment(
            physics, self._task, time_limit=_DEFAULT_TIME_LIMIT)
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
            new_physics = PusherPhysics.from_pomdp(pomdp_descriptor)
            self._env = control.Environment(
                new_physics, self._task, time_limit=_DEFAULT_TIME_LIMIT)

            if self._viewer:
                self._viewer.close()
                self._viewer = None

    @staticmethod
    def _physics_changed(old, new):
        return not (np.array_equal(old['goal'], new['goal'])
                    and old['gear'] == new['gear']
                    and old['obstacle_radius'] == new['obstacle_radius'])


class PusherPhysics(mjcf.Physics):
    @classmethod
    def get_mjcf(cls):
        m = mjcf.from_xml_string(*get_model_and_assets())

        # Change engineer mass so that it can be pushed
        m.worldbody.body['engineer'].geom[0].mass = 0.25

        # Add a joint/actuator/motor to allow 2D motion
        ballx = m.worldbody.body['boss'].add(
            'joint',
            name='ballx',
            type='slide',
            axis=[1, 0, 0],
            limited='false',
            damping=1.0)
        ballax = m.actuator.add('motor', name='ballax', gear=[50], joint=ballx)
        m.sensor.add('jointpos', name='ballx', joint=ballx)

        return m

    @classmethod
    def from_pomdp(cls, pomdp_descriptor):
        m = cls.get_mjcf()

        # Add a target geom
        mat = m.asset.add('material', name='target', rgba=[0.6, 0.3, 0.3, 1])
        target_pos = np.array([0., 0., 0.3])
        target_pos[:2] = pomdp_descriptor['goal']
        m.worldbody.add(
            'geom', name='target', pos=target_pos, material=mat, size='0.03')

        # Actuator gearing
        m.actuator.motor['ballax'].gear = [pomdp_descriptor['gear']]
        m.actuator.motor['ballay'].gear = [pomdp_descriptor['gear']]

        # Cylindrical obstacle
        mat = m.asset.add(
            'material', name='decoration', rgba=[0.3, 0.5, 0.7, 1.0])
        if pomdp_descriptor['obstacle_radius'] != 0:
            x, y, r = 0., -0.69, pomdp_descriptor['obstacle_radius']
            m.worldbody.add(
                'geom',
                name='obstacle',
                type='cylinder',
                size='{} {}'.format(float(r), 0.1),
                pos='{} {} {}'.format(float(x), float(y), 0.0),
                material=mat)

        return cls.from_mjcf_model(m)

    def engineer_to_target(self):
        """Returns the vector from engineer to target in global coordinate."""
        return (self.named.data.geom_xpos['target'][:2] -
                self.named.data.xpos['engineer'][:2])

    def engineer_to_target_dist(self):
        """Returns the distance from mass to the target."""
        return np.linalg.norm(self.engineer_to_target())


class PusherTask(base.Task):
    def __init__(self, random=None):
        super().__init__(random=random)
        self._observation_noise_std = 0.

    def initialize_episode(self, physics):
        pass

    def get_reward(self, physics):
        target_size = physics.named.model.geom_size['target', 0]
        return -physics.engineer_to_target_dist()

    def get_observation(self, physics):
        pos = physics.position()
        pos += np.random.normal(0, self._observation_noise_std, size=pos.shape)
        vel = physics.velocity()
        vel += np.random.normal(0, self._observation_noise_std, size=vel.shape)

        obs = collections.OrderedDict(position=pos, velocity=vel)
        return obs

    @property
    def observation_noise_std(self):
        return self._observation_noise_std

    @observation_noise_std.setter
    def observation_noise_std(self, std):
        self._observation_noise_std = std
