"""Device and determinism: the same seed gives the same numbers on the same device and versions."""

from __future__ import annotations

import os
import random
from typing import Any, Dict, Tuple

import numpy as np

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")   # deterministic cuBLAS; must be set before CUDA starts

import torch  # noqa: E402


def setup(seed: int, device: str = "auto") -> Tuple[torch.device, Dict[str, Any]]:
    """Seed every generator, ask for deterministic algorithms, pick the device. Returns it and a record of it."""

    random.seed(int(seed))
    np.random.seed(int(seed) % (2 ** 32))
    torch.manual_seed(int(seed))
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    chosen = torch.device(device)
    if chosen.type == "cuda":
        torch.cuda.manual_seed_all(int(seed))
    info = {"device": str(chosen), "torch": torch.__version__, "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(chosen) if chosen.type == "cuda" else None,
            "deterministic": True, "seed": int(seed)}
    return chosen, info
