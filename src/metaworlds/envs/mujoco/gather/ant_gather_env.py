from metaworlds.envs.mujoco import AntEnv
from metaworlds.envs.mujoco.gather import GatherEnv


class AntGatherEnv(GatherEnv):

    MODEL_CLASS = AntEnv
    ORI_IND = 6
