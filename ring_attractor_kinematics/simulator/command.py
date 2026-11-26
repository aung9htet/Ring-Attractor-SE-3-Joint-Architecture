"""``python -m ring_attractor_kinematics simulator``: give the hand a pose goal in a browser and watch the arm reach it.

    docker/run.bash sim -- 'python3 -B -m ring_attractor_kinematics simulator --chain tiago7'          then open http://localhost:8765/

With Gazebo as the robot (TIAGo in an empty world; ``--gpu`` for the camera picture)::

    docker/run.bash gazebo up ik-gazebo --gpu
    docker/run.bash gazebo exec ik-gazebo -- 'python3 -B -m ring_attractor_kinematics simulator --chain tiago7 --gazebo'
    docker/run.bash gazebo down ik-gazebo

``--stub`` replaces the spiking network by the exact rule (no NEST needed); ``--ticks N`` stops after N ticks.
The NEST and ROS back ends are loaded by name (:data:`BACKENDS`) only when asked for, so ``--stub`` runs anywhere.
"""

from __future__ import annotations

import argparse
import importlib
import os

from reaching_benchmark.geometry.chains import MODEL_NAMES

from ..config import POSITION_GOALS, TASKS
from .loop import METHODS, Simulator
from .server import SimServer, SimState
from .stub import StubNetwork

BACKENDS = {"spiking": "ring_attractor_kinematics.simulator.spiking", "gazebo": "reaching_benchmark.plants.gazebo_robot"}


def add_arguments(parser: argparse.ArgumentParser, command: str) -> None:
    parser.add_argument("--chain", default="tiago4", choices=MODEL_NAMES)
    parser.add_argument("--task", default=None, choices=TASKS, help="default: pose from six joints on, else position")
    parser.add_argument("--position-goal", default="sheets", choices=POSITION_GOALS)
    parser.add_argument("--method", default="ring", choices=METHODS)
    parser.add_argument("--gazebo", action="store_true", help="a Gazebo with the TIAGo runs beside this process: offer it as the robot")
    parser.add_argument("--port", type=int, default=int(os.environ.get("SIM_PORT", "8765")))
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--threads", type=int, default=12)
    parser.add_argument("--stub", action="store_true", help="the exact rule instead of the spiking network")
    parser.add_argument("--ticks", type=int, default=None, help="stop after this many ticks")


def main(args: argparse.Namespace, command: str) -> int:
    if args.stub:
        make_network = StubNetwork
    else:
        make_network = importlib.import_module(BACKENDS["spiking"]).network_factory(args.threads)
    state = SimState()
    make_robot, robots = None, ("light",)
    if args.gazebo:
        gazebo = importlib.import_module(BACKENDS["gazebo"])
        link = gazebo.RosLink()
        make_robot = lambda name, chain: gazebo.GazeboRobot(chain, link)  # noqa: E731
        robots = ("light", "gazebo")
        state.image = link.jpeg
    robot = "gazebo" if args.gazebo and args.chain.startswith("tiago") else "light"
    server = SimServer(state, args.host, args.port)
    server.start()
    simulator = Simulator(make_network, args.chain, state, task=args.task, position_goal=args.position_goal, method=args.method,
                          make_robot=make_robot, robots=robots, robot=robot)
    print("simulator ready: http://localhost:%d/  (chain %s, task %s, robot %s; Ctrl-C to stop)" % (
        server.port, args.chain, simulator.design.task, simulator.robot_name), flush=True)
    try:
        simulator.run_loop(args.ticks)
    except KeyboardInterrupt:
        pass
    finally:
        server.stop()
    return 0
