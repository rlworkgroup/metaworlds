from metaworlds.envs.mujoco import PointEnv
from metaworlds.envs.mujoco.gather import GatherEnv


class PointGatherEnv(GatherEnv):

    MODEL_CLASS = PointEnv
    ORI_IND = 2
