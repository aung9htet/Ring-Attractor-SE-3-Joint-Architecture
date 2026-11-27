"""Record the spikes drawn in Fig. 2a-c: a forward-only replay of one stored reach, not a benchmark (needs NEST).

The joint path of the RAK model's kinematic reach to TIAGo7 target pair 0 (kernel seed 1; the lowest index, fixed
before inspection) is replayed through the forward network (kernel seed 11), 2 s at the start pose and then 8 s of the
path, one 50 ms tick at a time. Each tick keeps the joint-ring spike counts, the three position sheets' spike counts, the
decoded ring centres and the sheet-decoded hand position. Writes ``analysis/source_data/illustrative_activity.npz`` and
its metadata. Takes a few seconds.
"""

from __future__ import annotations

import hashlib
import json

import numpy as np

from reaching_benchmark.geometry.kinematics import position

from .common import DATA, RAK, ensure_dirs, relative, write_json

SOURCE = RAK / "reach_kinematic" / "tiago7_seed1.json"
REPLAY_SEED = 11
THREADS = 8


def main() -> None:
    import nest

    from ring_attractor_kinematics.network.factory import build_forward

    ensure_dirs()
    doc = json.loads(SOURCE.read_text())
    row = next(r for r in doc["rows"] if r["index"] == 0)
    q_raw, t_raw = np.asarray(row["q"]), np.asarray(row["t"])
    chain, net = build_forward(nest, "tiago7", threads=THREADS, seed=REPLAY_SEED)
    nest.set_verbosity("M_ERROR")
    for _ in range(40):                       # 2 s at the start pose
        net.step(q_raw[0])
    times = np.arange(0, 8.0 + 1e-8, 0.05)
    q = np.column_stack([np.interp(times, t_raw, q_raw[:, k]) for k in range(chain.n)])
    counts, sheets, held, decoded, truth, ring_target = [], [], [], [], [], []
    for i, qi in enumerate(q):
        tick = net.step(qi)
        counts.append(net.ring_counts().copy())
        sheets.append(tick.sheet_counts.copy())
        held.append(tick.held.copy())
        decoded.append(tick.position.copy())
        truth.append(position(chain, qi))
        ring_target.append([r.mapping.position(a) for r, a in zip(net.rings, qi)])
        if i % 40 == 0:
            print("recorded %.2f s" % times[i], flush=True)
    np.savez_compressed(DATA / "illustrative_activity.npz", t=times, q=q, ring_counts=np.asarray(counts),
                        sheet_counts=np.asarray(sheets), held=np.asarray(held), decoded_position=np.asarray(decoded),
                        true_position=np.asarray(truth), ring_target=np.asarray(ring_target), sheet_places=net.mapping.places)
    write_json(DATA / "illustrative_activity_metadata.json", {
        "purpose": "illustrative forward replay, not a benchmark", "arm": "tiago7",
        "selection": "target pair 0 (lowest index) of the RAK model's kinematic reach, kernel seed 1; fixed before inspection",
        "source": relative(SOURCE), "source_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        "replay_network": "ForwardNetwork", "replay_kernel_seed": REPLAY_SEED, "threads": THREADS, "tick_ms": 50,
        "warmup_s": 2.0, "replay_s": 8.0, "time_points": len(times),
        "interpolation": "linear interpolation of the stored joint path (stored every 200 ms)",
        "neurons": net.neuron_count()})
    print("activity written to", relative(DATA / "illustrative_activity.npz"))
