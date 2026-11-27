#!/usr/bin/env bash
# Every run of the TIAGo arm in Gazebo that the baseline rows of Fig. 3b need, one after another, in one command:
#
#   nohup docker/run_tiago_baselines.bash > /dev/null 2>&1 &      (start it in the background)
#   tail -f logs/tiago_baselines.log                              (watch it: one line per trial, a heartbeat every minute)
#   python3 -m reaching_benchmark status --watch 60               (or every run at once, with its bar)
#
# 1. DLS IK and CCD IK on the 30 reported pairs (model image)                      results/dls_ik, results/ccd_ik
# 2. NN dynamics + MPC and PETS: collect, train (10 iterations), reach (baselines    results/mpc_nn_dynamics, results/pets
#    image, GPU)
# 3. every model side by side                                                       results/comparison/reach_gazebo.json
#
# Every step resumes where it stopped, so running this again after a stop goes on from there. A step that fails is
# reported in the log and the next step runs. The Gazebo containers are stopped at the end, or when this script stops.
set -uo pipefail
repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
cd "$repo"
mkdir -p logs
log=logs/tiago_baselines.log
exec >> "$log" 2>&1
export REACHING_PROGRESS=lines
cpu=tiago-baselines
gpu=tiago-baselines-gpu
stop() { docker/run.bash gazebo down "$cpu" > /dev/null 2>&1; docker/run.bash gazebo down "$gpu" > /dev/null 2>&1; true; }
trap 'echo "== $(date "+%F %T") stopped"; stop' EXIT
say() { echo "== $(date "+%F %T") $*"; }
step() {   # step CONTAINER COMMAND
    say "start: $2"
    if docker/run.bash gazebo exec "$1" -- "$2"; then say "done: $2"; else say "FAILED: $2 (see above); going on"; fi
}
stop
say "TIAGo baselines: DLS IK, CCD IK, NN dynamics + MPC, PETS on the 30 reported pairs in Gazebo"
say "Gazebo (model image)"
docker/run.bash gazebo up "$cpu" || { say "FAILED to start Gazebo"; exit 1; }
step "$cpu" 'python3 -B -m dls_ik reach --plant gazebo'
step "$cpu" 'python3 -B -m ccd_ik reach --plant gazebo'
docker/run.bash gazebo down "$cpu"
say "Gazebo (baselines image, GPU)"
docker/run.bash gazebo up "$gpu" --image baselines --cuda || { say "FAILED to start Gazebo with the GPU"; exit 1; }
step "$gpu" 'python3 -B -m mpc_nn_dynamics all --plant gazebo'
step "$gpu" 'python3 -B -m pets all --plant gazebo'
docker/run.bash gazebo down "$gpu"
say "comparison"
docker/run.bash nest -- 'python3 -B -m reaching_benchmark compare reach_gazebo'
say "all done"
