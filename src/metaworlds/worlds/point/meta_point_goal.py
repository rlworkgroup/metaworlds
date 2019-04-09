import numpy as np

from point_goal_world import PointGoalWorld
from world.meta import UniformRandomMetaWorld
import util

if __name__ == '__main__':
    # Create an instance of the world
    world = PointGoalWorld()
    util.print_running(world)

    # Create a metaworld to control the world
    meta = UniformRandomMetaWorld(world.pomdp_space)
    world.pomdp = meta.reset()
    util.print_running(meta)
    util.print_pomdp(world.pomdp)

    for _ in range(5):
        # Reset and step the POMDP in the World
        initial_state = world.reset()
        util.print_reset(initial_state)
        for _ in range(10):
            step = world.step({'velocity': np.array([0.1, 0.1])})
            util.print_step(step)

        # Sample a new POMDP
        pomdp, _, _, _ = meta.step(None)
        world.pomdp = pomdp
        util.print_pomdp(pomdp)