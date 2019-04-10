#!/usr/bin/env python3
import pprint
import numpy as np

from metaworlds.benchmarks import PusherSuite


def step_env(env):
    env.reset()
    for c in np.linspace(0., 0.6, 100):
        a = [-1., 0.]
        env.step(a)
        env.render()


try:
    print("Press Ctrl-C to stop...")
    world = PusherSuite.world()
    for b in PusherSuite.all_benchmarks:
        for pomdp in b.yield_test_set(4):
            print('Running {}'.format(pprint.pformat(pomdp)))
            world.pomdp = pomdp
            step_env(world)
    world.close()
except KeyboardInterrupt:
    print("Exiting...")
    world.close()
