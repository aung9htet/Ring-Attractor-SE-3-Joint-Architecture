#!/usr/bin/env bash
# Every controller on the TIAGo arm in Gazebo with one joint stuck from the start of each of the 30 reaches (Fig. 5d), in one
# command, using the machine fully:
#
#   nohup docker/run_tiago_stuck.bash > /dev/null 2>&1 &         (start it in the background)
#   tail -f logs/stuck_*.log                                      (watch it: one line per trial, a heartbeat every minute)
#   python3 -m reaching_benchmark status --watch 60               (or every run at once, with its bar)
#
# The stuck joint is each pair's pre-assigned one (the RAK model's joint-loss joints); from the start of the reach its
# trajectory target stays at its start angle, whatever the controller asks, and the controller is not told. Sticking it
# from the start means no controller has done part of the reach before it sticks. Each is compared with its own
# stored intact Gazebo run (results/<model>/reach_gazebo/). Results go to results/<model>/joint_loss_gazebo/.
#
#   phase 1, at the same time:  RAK model, RAK with predictive encoding   each over SHARDS Gazebos   logs/stuck_rak*.log
#                               DLS IK then CCD IK                        one Gazebo                 logs/stuck_reactive.log
#   phase 2, at the same time:  NN dynamics + MPC, PETS (stored models)   each over SHARDS Gazebos   logs/stuck_<model>.log
#
# Every step resumes where it stopped, so running this again after a stop goes on from there. All its Gazebo containers
# (named tiago-st-*) are stopped at the end, or when this script stops.
set -uo pipefail
repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
cd "$repo"
mkdir -p logs
SHARDS=${SHARDS:-3}
THREADS=${THREADS:-3}
export REACHING_PROGRESS=lines
exec >> logs/stuck.log 2>&1

say() { echo "== $(date "+%F %T") $*"; }
stop_all() {
    local names
    names=$(docker ps -a --format '{{.Names}}' | grep '^tiago-st-' || true)
    for name in $names; do docker/run.bash gazebo down "$name" > /dev/null 2>&1 || true; done
}
trap 'say "stopped"; stop_all' EXIT
up() {     # up NAME PORT [options]
    local name=$1 port=$2; shift 2
    say "Gazebo $name (port $port) $*"
    docker/run.bash gazebo up "$name" --port "$port" "$@" || { say "FAILED to start Gazebo $name"; return 1; }
}
down() { docker/run.bash gazebo down "$1" > /dev/null 2>&1 || true; }
step() {   # step NAME COMMAND
    say "start [$1]: $2"
    if docker/run.bash gazebo exec "$1" -- "$2"; then say "done [$1]: $2"; else say "FAILED [$1]: $2"; return 1; fi
}

sharded() {   # sharded TAG BASE_PORT COMMAND [gazebo options]: COMMAND --shard s/SHARDS in SHARDS Gazebos at once
    local tag=$1 base=$2 command=$3 s; shift 3
    for s in $(seq 0 $((SHARDS - 1))); do
        (
            name=tiago-st-$tag-$s
            up "$name" $((base + s)) "$@" && step "$name" "$command --shard $s/$SHARDS"
            down "$name"
        ) &
    done
    wait
}

rak() {       # rak TAG BASE_PORT [--variant V]
    local tag=$1 base=$2; shift 2
    sharded "$tag" "$base" "python3 -B -m ring_attractor_kinematics benchmark joint_loss_gazebo --condition lock --onset 0 --seed 1 --threads $THREADS $*"
}

reactive() {
    local name=tiago-st-reactive
    up "$name" 8890 || return 1
    step "$name" 'python3 -B -m dls_ik reach --plant gazebo --stuck'
    step "$name" 'python3 -B -m ccd_ik reach --plant gazebo --stuck'
    down "$name"
}

learned() {   # learned MODEL BASE_PORT
    local model=$1 base=$2
    sharded "$model" "$base" "python3 -B -m $model reach --plant gazebo --stuck" --image baselines --cuda
    say "summary [$model]"
    docker/run.bash nest --image baselines -- "python3 -B -m $model summarise --plant gazebo --stuck"
}

stop_all
say "phase 1: RAK model and RAK with predictive encoding ($SHARDS Gazebos each, $THREADS NEST threads), DLS IK and CCD IK"
rak rak 8860 >> logs/stuck_rak.log 2>&1 &
rak rak-pe 8870 --variant predictive_encoding >> logs/stuck_rak_pe.log 2>&1 &
reactive >> logs/stuck_reactive.log 2>&1 &
wait
docker/run.bash nest -- 'python3 -B -m ring_attractor_kinematics summarise joint_loss_gazebo'
say "phase 2: NN dynamics + MPC and PETS ($SHARDS Gazebos each)"
learned mpc_nn_dynamics 8900 >> logs/stuck_mpc_nn_dynamics.log 2>&1 &
learned pets 8910 >> logs/stuck_pets.log 2>&1 &
wait
say "figure"
docker/run.bash nest -- 'python3 -B -m analysis figure fig5'
say "all done"
