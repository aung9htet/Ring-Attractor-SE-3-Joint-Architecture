#!/usr/bin/env bash
# Containers for this repo: tests, calibration, experiments, reports and the simulator.
#
#   docker/run.bash nest [--out DIR] [--so2 DIR] -- COMMAND     one-shot, no network
#   docker/run.bash sim  [--port 8765] [--out DIR] -- COMMAND   the same, with one port published on localhost
#
# COMMAND is one shell string.  It runs in /work (this repo, mounted read-write: calibration
# writes artifacts/, reports write docs/results/) with NEST and this package on the Python path.
# --out DIR is mounted at /out.  --so2 DIR mounts a checkout of the SO-2 repo read-only at /so2
# (only scripts/run_old_decoder.py needs it).  The image has no pytest; tests use unittest:
#
#   docker/run.bash nest -- 'python3 -B -m unittest discover -s tests'
#   docker/run.bash sim -- 'python3 -B scripts/simulator.py --chain tiago7'
set -euo pipefail
repo_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
source "$repo_dir/docker/image.env"
setup='source /opt/ros/noetic/setup.bash; source /tiago_public_ws/devel/setup.bash; source /usr/local/nest/bin/nest_vars.sh; export PYTHONPATH=/work:${PYTHONPATH:-}; cd /work'
banner='^\s*$|-- N E S T --|Copyright \(C\) 2004|Version: HEAD|Built: |This program is provided|NO WARRANTY|Problems or suggestions|Visit https://www.nest|Type .nest.help'
die() { echo "Error: $*" >&2; exit 1; }
usage() { sed -n '2,13p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; }

mode=${1:-}; shift || true
case "$mode" in nest|sim) ;; -h|--help|'') usage; exit 0 ;; *) die "unknown mode $mode (nest, sim)" ;; esac
out=; so2=; port=8765
while (($#)); do
    case "$1" in
        --out) out=$2; shift 2 ;;
        --so2) so2=$2; shift 2 ;;
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
[[ -z $so2 ]] || { [[ -d $so2/src/tiago_ring_controller ]] || die "$so2 is not a checkout of the SO-2 repo"; args+=(-v "$(cd "$so2" && pwd -P):/so2:ro"); }
if [[ $mode == sim ]]; then
    args+=(-p "127.0.0.1:$port:$port" -e "SIM_PORT=$port")
else
    args+=(--network none)
fi
docker run "${args[@]}" --entrypoint bash "$MODEL_IMAGE" -c "$setup && $1" 2>&1 | grep --line-buffered -v -E "$banner" || true
