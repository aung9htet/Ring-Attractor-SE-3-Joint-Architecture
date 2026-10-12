#!/usr/bin/env python3
"""The IK simulator: give the hand a pose in a browser and watch the arm go to it.

    docker/run.bash sim -- 'python3 -B scripts/ik_simulator.py --chain tiago4'
    then open http://localhost:8765/

With Gazebo as the robot (the TIAGo in an empty world; --gpu for the camera picture):

    docker/run.bash gazebo up ik-gazebo --gpu
    docker/run.bash gazebo exec ik-gazebo -- 'python3 -B scripts/ik_simulator.py --chain tiago7 --gazebo'
    docker/run.bash gazebo down ik-gazebo

The page shows the arm with the goal (drag its dot or an arrow tip to move it, a ring to turn it),
the task (pose, position only, orientation only), where positions are compared (on the sheets, or
as numbers), a choice of methods (the ring IK; the exact cascade; damped least squares and Jacobian
transpose as standard methods), the robot (light robot or Gazebo), the distance from the goal for
the last runs side by side, the goal's and the hand's packets on the three sheets, every joint's
share, and the spikes of all rings.

``tiago4`` is asked for the position (4 joints); ``generic7`` and ``tiago7`` for the whole pose
(about 70,000 neurons: they run at a fifth of real time).  ``--stub`` replaces the spiking network
by the exact rule, to try the page where NEST is not installed.  ``--ticks N`` stops after N ticks.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from ring_se3.chains import MODEL_NAMES  # noqa: E402
from ring_se3.ik.design import POSITION_GOALS, TASKS  # noqa: E402
from ring_se3.sim.ik_server import METHODS, PoseIkSimServer, PoseIkSimState, PoseIkSimulator, PoseIkStubNetwork  # noqa: E402


def network_factory(threads):
    import nest
    from ring_so3.design import Design
    from ring_so3.nest import calibrate as so3_calibrate
    from ring_so3.nest import ik_calibrate as so3_ik_calibrate
    from ring_se3.nest import calibrate, ik_calibrate
    from ring_se3.nest.ik_network import PoseIkNetwork, ik_options

    design = Design()
    base, pose, ik, ramp = so3_calibrate.load(design), calibrate.load(design), so3_ik_calibrate.load(design), ik_calibrate.load(design)
    return lambda chain, ik_design: PoseIkNetwork(nest, chain, base, pose, ik, ramp, design, ik_options(threads=threads), ik_design)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--chain", default="tiago4", choices=MODEL_NAMES)
    parser.add_argument("--task", default=None, choices=TASKS, help="default: pose from six joints on, else position")
    parser.add_argument("--position-goal", default="sheets", choices=POSITION_GOALS)
    parser.add_argument("--method", default="ring", choices=METHODS)
    parser.add_argument("--gazebo", action="store_true", help="a Gazebo with the TIAGo runs beside this process: offer it as the robot")
    parser.add_argument("--robot", default=None, choices=("light", "gazebo"), help="default: gazebo with --gazebo and a TIAGo chain, else light")
    parser.add_argument("--port", type=int, default=int(os.environ.get("SIM_PORT", "8765")))
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--threads", type=int, default=12)
    parser.add_argument("--stub", action="store_true", help="the exact rule instead of the spiking network")
    parser.add_argument("--ticks", type=int, default=None, help="stop after this many ticks")
    args = parser.parse_args(argv)

    make_network = (lambda chain, ik_design: PoseIkStubNetwork(chain, ik_design)) if args.stub else network_factory(args.threads)
    state = PoseIkSimState()
    make_robot, robots = None, ("light",)
    if args.gazebo:
        from ring_se3.gazebo.robot import PoseGazeboRobot as GazeboRobot
        from ring_se3.gazebo.link import PoseRosLink

        link = PoseRosLink()
        make_robot = lambda name, chain: GazeboRobot(chain, link)  # noqa: E731
        robots = ("light", "gazebo")
        state.image = link.jpeg
    robot = args.robot or ("gazebo" if args.gazebo and args.chain.startswith("tiago") else "light")
    server = PoseIkSimServer(state, args.host, args.port)
    server.start()
    simulator = PoseIkSimulator(make_network, args.chain, state, task=args.task, position_goal=args.position_goal, method=args.method,
                                make_robot=make_robot, robots=robots, robot=robot)
    print("IK simulator ready: http://localhost:%d/  (chain %s, task %s, robot %s, Ctrl-C to stop)" % (
        server.port, args.chain, simulator.design.task, simulator.robot_name), flush=True)
    try:
        simulator.run_loop(args.ticks)
    except KeyboardInterrupt:
        pass
    finally:
        server.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
