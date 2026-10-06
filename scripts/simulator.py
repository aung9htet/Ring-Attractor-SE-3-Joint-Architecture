#!/usr/bin/env python3
"""The simulator: change the joint angles in a browser and watch the pose read from the network next to the true one.

    docker/run.bash sim -- 'python3 -B scripts/simulator.py --chain tiago4'
    then open http://localhost:8765/

The page has a slider per joint, a mode in which the arm keeps moving from one random goal
to the next, a 3D view of the arm with the true and the decoded end point and frames,
position and orientation over time, the three attractor sheets, and switches for the anchor
and for path integration.  The spiking network runs live, one 50 ms tick at a time, in step
with the robot; the status line says how fast (the 7-joint network runs at well under real
time, a 3- or 4-joint one close to it).

``--stub`` replaces the network by exact kinematics with a little noise, to try the page
where NEST is not installed.  ``--ticks N`` stops after N ticks.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from ring_se3.chains import MODEL_NAMES  # noqa: E402
from ring_se3.sim.server import PoseSimServer, PoseSimState, PoseSimulator, PoseStubNetwork  # noqa: E402


def network_factory(threads, velocity=True):
    import nest
    from ring_so3.design import Design
    from ring_so3.nest import calibrate as so3_calibrate
    from ring_se3.nest import calibrate
    from ring_se3.nest.network import PoseNetwork, PoseOptions

    design = Design()
    base, pose = so3_calibrate.load(design), calibrate.load(design)
    return lambda chain: PoseNetwork(nest, chain, base, pose, design, PoseOptions(threads=threads, velocity=velocity))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--chain", default="tiago4", choices=MODEL_NAMES)
    parser.add_argument("--port", type=int, default=int(os.environ.get("SIM_PORT", "8765")))
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--threads", type=int, default=12)
    parser.add_argument("--speed", type=float, default=0.25, help="joint speed limit at the start, rad/s")
    parser.add_argument("--mode", choices=("manual", "goals"), default="goals")
    parser.add_argument("--lead", action="store_true", help="start with 'write ahead' on (the page has a switch for it)")
    parser.add_argument("--stub", action="store_true", help="exact kinematics instead of the spiking network")
    parser.add_argument("--ticks", type=int, default=None, help="stop after this many ticks")
    args = parser.parse_args(argv)

    make_network = (lambda chain: PoseStubNetwork(chain)) if args.stub else network_factory(args.threads)
    state = PoseSimState()
    server = PoseSimServer(state, args.host, args.port)
    server.start()
    simulator = PoseSimulator(make_network, args.chain, state, speed=args.speed, mode=args.mode, lead=args.lead)
    print("simulator ready: http://localhost:%d/  (chain %s, Ctrl-C to stop)" % (server.port, args.chain), flush=True)
    try:
        simulator.run(args.ticks)
    except KeyboardInterrupt:
        pass
    finally:
        server.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
