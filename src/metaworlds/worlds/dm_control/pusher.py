import collections

from dm_control import mujoco
from dm_control.rl import control
from dm_control.suite import base
from dm_control.utils import rewards
import numpy as np

from metaworlds import pallet

_DEFAULT_TIME_LIMIT = 20


def get_model_and_assets():
    """Returns a tuple containing the model XML string and a dict of assets."""
    return pallet.read_model('pusher/pusher.xml'), pallet.ASSETS


def pusher(time_limit=_DEFAULT_TIME_LIMIT,
           random=None,
           environment_kwargs=None):
    """Returns the pusher task."""
    physics = Physics.from_xml_string(*get_model_and_assets())
    task = Pusher(random=random)
    environment_kwargs = environment_kwargs or {}
    return control.Environment(
        physics, task, time_limit=time_limit, **environment_kwargs)


class Physics(mujoco.Physics):
    """physics for the point_mass domain."""

    TARGET = np.array([0., -1., 0.12])

    def engineer_to_target(self):
        """Returns the vector from engineer to target in global coordinate."""
        return (Physics.TARGET - self.named.data.xpos['engineer'])

    def engineer_to_target_dist(self):
        """Returns the distance from mass to the target."""
        return np.linalg.norm(self.engineer_to_target())


class Pusher(base.Task):
    def __init__(self, random=None):
        super().__init__(random=random)

    def initialize_episode(self, physics):
        pass

    def get_reward(self, physics):
        """Replace dm_control's shaped reward with a simple L2 norm cost"""
        ball_size = physics.named.model.geom_size['ball', 0]
        near_target = rewards.tolerance(
            physics.engineer_to_target_dist(),
            bounds=(0, ball_size),
            margin=ball_size)
        return near_target

    def get_observation(self, physics):
        """Returns an observation of the state."""
        obs = collections.OrderedDict()
        obs['position'] = physics.position()
        obs['velocity'] = physics.velocity()
        return obs
