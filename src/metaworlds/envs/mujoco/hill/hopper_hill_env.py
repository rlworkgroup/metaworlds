import gym
import numpy as np

from metaworlds.envs.mujoco import HopperEnv
from metaworlds.envs.mujoco.hill import HillEnv
from metaworlds.envs.mujoco.hill import terrain
from metaworlds.misc.overrides import overrides


class HopperHillEnv(HillEnv):

    MODEL_CLASS = HopperEnv

    @overrides
    def _mod_hfield(self, hfield):
        # clear a flat patch for the robot to start off from
        return terrain.clear_patch(
            hfield,
            gym.spaces.Box(
                np.array([-1.0, -1.0]),
                np.array([-0.5, -0.5]),
                dtype=np.float32))
