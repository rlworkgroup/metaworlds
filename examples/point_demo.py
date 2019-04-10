#!/usr/bin/env python3
import pprint
import numpy as np

from metaworlds.benchmarks import PointMassSuite


def step_env(env):
    env.reset()
    for c in np.linspace(0., 1., 100):
        a = c * env.action_space.high + (1 - c) * env.action_space.low
        env.step(a)
        env.render()


try:
    print("Press Ctrl-C to stop...")
    world = PointMassSuite.world()
    for b in PointMassSuite.all_benchmarks:
        for pomdp in b.yield_test_set(4):
            print('Running {}'.format(pprint.pformat(pomdp)))
            world.pomdp = pomdp
            step_env(world)
    world.close()

except KeyboardInterrupt:
    print("Exiting...")
    world.close()
