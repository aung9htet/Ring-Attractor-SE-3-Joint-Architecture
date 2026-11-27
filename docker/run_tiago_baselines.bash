#!/usr/bin/env bash
# Every run of the TIAGo arm in Gazebo that the baseline rows of Fig. 3b need, in one command, using the machine fully:
#
#   nohup docker/run_tiago_baselines.bash > /dev/null 2>&1 &      (start it in the background)
#   tail -f logs/tiago_*.log                                      (watch it: one line per trial or training episode,
#                                                                   a heartbeat every minute while one runs)
#   python3 -m reaching_benchmark status --watch 60               (or every run at once, with its bar)
#
# What runs at the same time, each in a Gazebo of its own (Gazebo's physics settings are unchanged, so every run is
# measured exactly as the RAK model's Gazebo runs were):
#
#   DLS IK, CCD IK         one Gazebo, one after the other (minutes)             log: logs/tiago_reactive.log
#   NN dynamics + MPC      collect and train in one Gazebo, then its 30 test     log: logs/tiago_mpc_nn_dynamics.log
#                          reaches split over SHARDS Gazebos (default 3)
#   PETS                   the same                                              log: logs/tiago_pets.log
#
# Training cannot be split (each round of phase 2 needs the model the last round made); the test reaches can (each pair is
# independent; the summary puts the shards back together). Then every model side by side (logs/tiago_baselines.log).
# Every step resumes where it stopped, so running this again after a stop goes on from there. All its Gazebo containers
# (named tiago-bl-*) are stopped at the end, or when this script stops.
set -uo pipefail
repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
cd "$repo"
mkdir -p logs
SHARDS=${SHARDS:-3}
export REACHING_PROGRESS=lines
exec >> logs/tiago_baselines.log 2>&1

say() { echo "== $(date "+%F %T") $*"; }
stop_all() {
    local names
    names=$(docker ps -a --format '{{.Names}}' | grep '^tiago-bl-' || true)
    for name in $names; do docker/run.bash gazebo down "$name" > /dev/null 2>&1 || true; done
}
trap 'say "stopped"; stop_all' EXIT
up() {     # up NAME PORT [options]: start a Gazebo, say so in the log
    local name=$1 port=$2; shift 2
    say "Gazebo $name (port $port) $*"
    docker/run.bash gazebo up "$name" --port "$port" "$@" || { say "FAILED to start Gazebo $name"; return 1; }
}
down() { docker/run.bash gazebo down "$1" > /dev/null 2>&1 || true; }
step() {   # step NAME COMMAND
    say "start [$1]: $2"
    if docker/run.bash gazebo exec "$1" -- "$2"; then say "done [$1]: $2"; else say "FAILED [$1]: $2"; return 1; fi
}

reactive() {
    local name=tiago-bl-reactive
    up "$name" 8790 || return 1
    step "$name" 'python3 -B -m dls_ik reach --plant gazebo'
    step "$name" 'python3 -B -m ccd_ik reach --plant gazebo'
    down "$name"
}

learned() {   # learned MODEL BASE_PORT
    local model=$1 base=$2 name=tiago-bl-$1 s
    up "$name" "$base" --image baselines --cuda || return 1
    if step "$name" "python3 -B -m $model collect --plant gazebo" && step "$name" "python3 -B -m $model train --plant gazebo"; then
        down "$name"
        for s in $(seq 0 $((SHARDS - 1))); do
            (
                shard=$name-$s
                up "$shard" $((base + 1 + s)) --image baselines --cuda && step "$shard" "python3 -B -m $model reach --plant gazebo --shard $s/$SHARDS"
                down "$shard"
            ) &
        done
        wait
        say "summary [$model]"
        docker/run.bash nest --image baselines -- "python3 -B -m $model summarise --plant gazebo"
    else
        down "$name"
        return 1
    fi
}

stop_all
say "TIAGo baselines in parallel: DLS IK and CCD IK; NN dynamics + MPC; PETS (test reaches over $SHARDS Gazebos each)"
reactive >> logs/tiago_reactive.log 2>&1 &
learned mpc_nn_dynamics 8770 >> logs/tiago_mpc_nn_dynamics.log 2>&1 &
learned pets 8780 >> logs/tiago_pets.log 2>&1 &
wait
say "comparison"
docker/run.bash nest -- 'python3 -B -m reaching_benchmark compare reach_gazebo'
say "all done"
