#!/usr/bin/env bash
# Containers for this repo: tests, calibration, benchmarks of every model, and the simulator.
#
#   docker/run.bash nest [--image baselines] [--cuda] [--out DIR] -- COMMAND        one-shot, no network
#   docker/run.bash sim  [--port 8765] [--out DIR] -- COMMAND                      the same, with one port published on localhost
#   docker/run.bash gazebo up NAME [--image baselines] [--cuda] [--gpu] [--port 8765] [--out DIR]
#                                                       start Gazebo with the TIAGo in an empty world, in the background
#   docker/run.bash gazebo exec NAME -- COMMAND        run a command beside that Gazebo (ROS, NEST and this repo set up)
#   docker/run.bash gazebo down NAME                   stop it
#   docker/run.bash build-baselines                    build the baselines image (PyTorch) and print its digest
#
# COMMAND is one shell string.  It runs in /work (this repo, mounted read-write: calibration writes
# ring_attractor_kinematics/artifacts/, benchmarks write results/) with NEST and every package of this repo on the
# Python path.  --out DIR is mounted at /out.
# --image baselines runs the baselines image (the model image plus PyTorch: docker/Dockerfile.baselines), which the
# learned baselines (mpc_nn_dynamics, pets) need; --cuda gives the container the NVIDIA GPUs for computing (PyTorch).
# Gazebo: the world is gazebo/ring_ik.world (ground, light, one camera); --gpu renders through the desktop's X server,
# which only the camera picture needs.  The port is the one a simulator started with "gazebo exec" will serve on.
# Progress bars of long runs go to stderr and redraw in place when this script's stderr is a terminal; with stderr
# redirected (nohup, a log file) they print one line per trial instead.  REACHING_PROGRESS=bar|lines|off overrides.
# The image has no pytest; tests use unittest:
#
#   docker/run.bash nest -- 'python3 -B -m unittest discover -s tests'
#   docker/run.bash sim -- 'python3 -B -m ring_attractor_kinematics simulator --chain tiago7'
set -euo pipefail
repo_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
source "$repo_dir/docker/image.env"
setup='source /opt/ros/noetic/setup.bash; source /tiago_public_ws/devel/setup.bash; source /usr/local/nest/bin/nest_vars.sh; export PYTHONPATH=/work:${PYTHONPATH:-}; cd /work'
banner='^\s*$|-- N E S T --|Copyright \(C\) 2004|Version: HEAD|Built: |This program is provided|NO WARRANTY|Problems or suggestions|Visit https://www.nest|Type .nest.help'
die() { echo "Error: $*" >&2; exit 1; }
usage() { sed -n '2,25p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; }
# how reaching_benchmark/progress.py draws: in place on a terminal, line by line otherwise (inside the container stderr is never one)
progress_mode() { if [[ -n ${REACHING_PROGRESS:-} ]]; then echo "$REACHING_PROGRESS"; elif [[ -t 2 ]]; then echo bar; else echo lines; fi; }

# the image of --image NAME: model (default) or baselines
image_of() { case "$1" in model) echo "$MODEL_IMAGE" ;; baselines) [[ -n ${BASELINE_IMAGE:-} ]] || die 'no BASELINE_IMAGE in docker/image.env: run "docker/run.bash build-baselines" first'; echo "$BASELINE_IMAGE" ;; *) die "unknown image $1 (model, baselines)" ;; esac; }

mode=${1:-}; shift || true
case "$mode" in nest|sim|gazebo) ;; build-baselines)
    docker build -f "$repo_dir/docker/Dockerfile.baselines" --build-arg "MODEL_IMAGE=$MODEL_IMAGE" -t ring-attractor-kinematics-baselines:torch2.4.1-cu121 "$repo_dir/docker"
    echo "built ring-attractor-kinematics-baselines:torch2.4.1-cu121, image id: $(docker image inspect -f '{{.Id}}' ring-attractor-kinematics-baselines:torch2.4.1-cu121)"
    echo "record it in docker/image.env as BASELINE_IMAGE=<that id>"
    exit 0 ;;
    -h|--help|'') usage; exit 0 ;; *) die "unknown mode $mode (nest, sim, gazebo, build-baselines)" ;; esac
if [[ $mode == gazebo ]]; then
    action=${1:-}; name=${2:-}; shift 2 || die 'gazebo needs: up NAME, exec NAME -- COMMAND, or down NAME'
    case "$action" in
    up)
        gpu=0; cuda=0; image=model; out=; port=8765
        while (($#)); do
            case "$1" in
                --gpu) gpu=1; shift ;;
                --cuda) cuda=1; shift ;;
                --image) image=$2; shift 2 ;;
                --port) port=$2; shift 2 ;;
                --out) out=$2; shift 2 ;;
                *) die "unknown option $1" ;;
            esac
        done
        [[ -z $(docker ps -aq -f "name=^${name}\$") ]] || die "a container named $name exists already: stop it with \"docker/run.bash gazebo down $name\", or pick another name"
        if (exec 3<>"/dev/tcp/127.0.0.1/$port") 2>/dev/null; then
            die "port $port is in use on this machine (another simulator?): add --port with a free one, e.g. --port $((port + 1)), (a simulator started in the container uses that port by itself)"
        fi
        args=(-d --rm --init --name "$name" --shm-size 1g --user "$(id -u):$(id -g)"
              -e HOME=/tmp -e USER=model -e LOGNAME=model -e PYTHONDONTWRITEBYTECODE=1 -e MPLBACKEND=Agg -e MPLCONFIGDIR=/tmp/mpl
              -e ROS_MASTER_URI=http://localhost:11311 -e ROS_HOSTNAME=localhost -e GAZEBO_MODEL_DATABASE_URI= -e "SIM_PORT=$port"
              -p "127.0.0.1:$port:$port" -v "$repo_dir:/work"
              -v "$repo_dir/gazebo/ring_ik.world:/tiago_public_ws/src/pal_gazebo_worlds/worlds/ring_ik.world:ro")
        [[ -z $out ]] || { mkdir -p "$out"; args+=(-v "$(cd "$out" && pwd -P):/out"); }
        if ((gpu)); then
            [[ -n ${DISPLAY:-} ]] || die '--gpu needs a desktop session (DISPLAY)'
            xauth_file=$(mktemp /tmp/ring-se3-xauth.XXXXXX)
            xauth nlist "$DISPLAY" | sed 's/^..../ffff/' | xauth -f "$xauth_file" nmerge -
            chmod 644 "$xauth_file"
            args+=(--gpus all -e NVIDIA_DRIVER_CAPABILITIES=graphics,utility,compute,display
                   -e __NV_PRIME_RENDER_OFFLOAD=1 -e __GLX_VENDOR_LIBRARY_NAME=nvidia
                   -e "DISPLAY=$DISPLAY" -e XAUTHORITY=/tmp/x.xauth -e QT_X11_NO_MITSHM=1
                   -v /tmp/.X11-unix:/tmp/.X11-unix:ro -v "$xauth_file:/tmp/x.xauth:ro")
        elif ((cuda)); then
            args+=(--gpus all -e NVIDIA_DRIVER_CAPABILITIES=compute,utility -e DISPLAY= -e XAUTHORITY= -e LIBGL_ALWAYS_SOFTWARE=1)
        else
            args+=(-e DISPLAY= -e XAUTHORITY= -e LIBGL_ALWAYS_SOFTWARE=1 -e NVIDIA_VISIBLE_DEVICES=void)
        fi
        docker run "${args[@]}" --entrypoint bash "$(image_of "$image")" -c "source /opt/ros/noetic/setup.bash; source /tiago_public_ws/devel/setup.bash; exec roslaunch tiago_gazebo tiago_gazebo.launch world:=ring_ik gui:=false public_sim:=true arm:=true end_effector:=pal-hey5 ft_sensor:=schunk-ft" >/dev/null
        # ready when the arm controller runs; the start-up motion of the arm then needs about 30 s more
        docker exec -i "$name" bash -c 'source /opt/ros/noetic/setup.bash; python3 -' <<'PY'
import subprocess, time
started = time.time()
while time.time() - started < 300:
    out = subprocess.run("rosservice call /controller_manager/list_controllers 2>/dev/null", shell=True, capture_output=True, text=True).stdout
    if "arm_controller" in out and out.count("running") >= 4:
        break
    time.sleep(3)
else:
    raise SystemExit("the arm controller did not come up in 300 s")
time.sleep(30)
print("Gazebo ready after %.0f s" % (time.time() - started))
PY
        echo "A simulator started in this container is served on http://localhost:$port/ . Stop everything with: docker/run.bash gazebo down $name"
        ;;
    exec)
        [[ ${1:-} == -- ]] && shift
        (($# == 1)) || die 'give one command string after --'
        [[ -n $(docker ps -q -f "name=^${name}\$") ]] || die "no running container named $name: start it with \"docker/run.bash gazebo up $name\""
        # with a terminal, Ctrl-C reaches the command inside the container (without one it would keep running there)
        tty=(); [[ -t 0 ]] && tty=(-t)
        # one stream here (a terminal merges them): progress comes line by line, which the banner filter passes as it comes
        docker exec -i "${tty[@]}" -e "REACHING_PROGRESS=${REACHING_PROGRESS:-lines}" "$name" bash -c "$setup && $1" 2>&1 \
            | grep --line-buffered -v -E "$banner" || true
        ;;
    down) docker stop "$name" >/dev/null && echo "stopped $name" ;;
    *) die "unknown gazebo action $action (up, exec, down)" ;;
    esac
    exit 0
fi
out=; port=8765; image=model; cuda=0
while (($#)); do
    case "$1" in
        --out) out=$2; shift 2 ;;
        --image) image=$2; shift 2 ;;
        --cuda) cuda=1; shift ;;
        --port) port=$2; shift 2 ;;
        --) shift; break ;;
        *) die "unknown option $1 (put -- before the command)" ;;
    esac
done
(($# == 1)) || die 'give one command string after --'
args=(--rm -i --init --shm-size 1g --user "$(id -u):$(id -g)"
      -e HOME=/tmp -e USER=model -e LOGNAME=model -e PYTHONDONTWRITEBYTECODE=1 -e MPLBACKEND=Agg -e MPLCONFIGDIR=/tmp/mpl
      -v "$repo_dir:/work")
[[ -z $out ]] || { mkdir -p "$out"; args+=(-v "$(cd "$out" && pwd -P):/out"); }
if [[ $mode == sim ]]; then
    args+=(-p "127.0.0.1:$port:$port" -e "SIM_PORT=$port")
else
    args+=(--network none)
fi
((cuda)) && args+=(--gpus all -e NVIDIA_DRIVER_CAPABILITIES=compute,utility)
args+=(-e "REACHING_PROGRESS=$(progress_mode)" -e "COLUMNS=${COLUMNS:-$(tput cols 2>/dev/null || echo 120)}")
# stdout goes through the banner filter; stderr (progress bars, tracebacks) goes straight to this script's stderr, so a
# bar can redraw in place
{ docker run "${args[@]}" --entrypoint bash "$(image_of "$image")" -c "$setup && $1" 2>&3 | grep --line-buffered -v -E "$banner" || true; } 3>&2
