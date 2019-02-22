from metaworlds.envs.mujoco import SwimmerEnv
from metaworlds.envs.mujoco.gather import GatherEnv


class SwimmerGatherEnv(GatherEnv):

    MODEL_CLASS = SwimmerEnv
    ORI_IND = 2
