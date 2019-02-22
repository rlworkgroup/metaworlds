import math
import unittest

from metaworlds.envs.mujoco.maze.maze_env_utils import line_intersect
from metaworlds.envs.mujoco.maze.maze_env_utils import ray_segment_intersect


class TestMazeEnv(unittest.TestCase):
    def test_line_intersect(self):
        assert line_intersect((0, 0), (0, 1), (0, 0), (1, 0))[:2] == (0, 0)
        assert line_intersect((0, 0), (0, 1), (0, 0), (0, 1))[2] == 0
        assert ray_segment_intersect(
            ray=((0, 0), 0), segment=((1, -1), (1, 1))) == (1, 0)
        assert ray_segment_intersect(
            ray=((0, 0), math.pi), segment=((1, -1), (1, 1))) is None
