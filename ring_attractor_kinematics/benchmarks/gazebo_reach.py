"""B6: closed-loop reaching with the TIAGo arm in Gazebo, and the joint-loss study there (``gazebo_joint_loss``). Run it
beside a Gazebo started with ``docker/run.bash gazebo up``."""

from __future__ import annotations

from typing import Any

import nest

from reaching_benchmark.plants.gazebo_robot import GazeboRobot, RosLink

from .reach import plan, run_no_spikes, run_one


def run_command(args: Any) -> None:
    link = RosLink()
    shard = tuple(int(v) for v in args.shard.split("/"))
    for run in plan(args.name, args):
        if run.get("method") == "cascade":
            run_no_spikes(args.name, run, (shard[0], shard[1]), args.max_minutes, make_robot=lambda chain: GazeboRobot(chain, link))
        else:
            run_one(nest, args.name, run, (shard[0], shard[1]), args.max_minutes, args.threads, make_robot=lambda chain: GazeboRobot(chain, link))
