"""``python -m pets {collect | train | reach | summarise | all} [options]``: see learned_dynamics/protocol.py."""

from learned_dynamics.protocol import main

from .config import CONFIG

if __name__ == "__main__":
    raise SystemExit(main(CONFIG))
