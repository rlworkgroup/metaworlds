#!/usr/bin/env python3
from garage.envs.dm_control import DmControlEnv
import numpy as np

from metaworlds.worlds.pusher import pusher


def step_env(env):
    env.reset()
    for c in np.linspace(0., 1., 100):
        a = c * env.action_space.high + (1 - c) * env.action_space.low
        env.step(a)
        env.render()


try:
    print("Press Ctrl-C to stop...")
    env = DmControlEnv(pusher())
    while True:
        step_env(env)

except KeyboardInterrupt:
    print("Exiting...")
    env.close()
