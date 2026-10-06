#!/usr/bin/env python3
"""The example of README.md, section 7: build a network once, step it, read the result.

    docker/run.bash nest -- 'python3 -B scripts/example.py'
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import nest
import numpy as np
from ring_se3 import experiments, kinematics

chain, design, network, session = experiments.build(nest, "tiago4", threads=8)   # built once, then kept
session.warm_up()                                       # 1 s: the rings and the packets form
measurement = session.go_to(chain.limits().mean(axis=1) + 0.3)       # move there, rest 1 s
print(measurement.position)                             # end point read from the sheets, m, base frame
print(kinematics.position(chain, session.robot.q))      # the exact one
tick = session.tick()                                   # one more 50 ms tick
print(tick.position, tick.cells_position)               # from the sheets (last two ticks), from the position cells
print(tick.tool_rotation())                             # orientation, as in the SO-3 repo
network.set_anchor(False)                               # the packets are on their own now
