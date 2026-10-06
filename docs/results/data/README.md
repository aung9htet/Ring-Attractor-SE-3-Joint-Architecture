# The data behind the results

Everything here was written by the scripts named below, in the container. `scripts/report.py`
reads it and writes [`../tables.md`](../tables.md) and [`../figures/`](../figures/). To redo
it: top-level [`README.md`](../../../README.md), section 3.7.

| Path | Test | Written by | Holds |
| --- | --- | --- | --- |
| `rate_study.json` | 2 | `scripts/rate_study.py` | position and velocity populations in the copy without spikes, every model |
| `parts.json` | 3 | `scripts/run_parts.py` | a packet left alone, an anchored sheet, a packet moved by a known velocity, the velocity populations of whole networks; the speed curve of the calibration |
| `static/<model>.json` | 4, 5, 6, 10 | `scripts/run_static.py --chain <model>` | 300 random held poses of one model |
| `tracking/<model>_<speed>.json` | 7 | `scripts/run_tracking.py` | every tick of a session in which the arm keeps moving |
| `tracking_variants/<model>_<variant>.json` | 7 | `scripts/run_tracking.py` with `--no-path-integration`, `--anchor-lead 0.2` or `--lead` | the same at 0.5 rad/s: `anchor_only`, `anchor_lead`, `joint_lead` |
| `dropout/<model>_<name>.json` | 8 | `scripts/run_dropout.py` | sessions in which the anchor is switched off and on: `rest` (arm at rest), `move010`, `move025`, `move050` (arm moving at that joint speed limit), each also with `_no_pi` (path integration off) |
| `trail/<model>_<mode>.json` | | `scripts/run_trail.py` | the shift-register experiment: `trail`, `none`, `velocity` |
| `robust/<model>_<variant>.json` | 9 | `scripts/run_static.py` with one option | 100 poses of a variant (names: `VARIANTS` in `scripts/run_experiments.py`) |
| `speed.json` | | `scripts/benchmark.py` | simulated time per wall-clock time of every model |
| `so3_reference.json` | 10 | copied numbers | orientation error of every model in the SO-3 repo (its `docs/results/data/static/`), for the comparison of test 10 |
| `logs/` | | `scripts/run_experiments.py` | the output of every run (not kept in git) |

Lengths are in metres, angles in radians unless a name says `deg`. `radius_m` is the arm's
reach; the tables give position errors as a share of it.

## A static file

Top level: the chain, the design, the sheet design, `radius_m` and `centre_m` of the
workspace, the options of the network, its size (`neurons`, `so3_neurons`, `products`,
`pose_products`, `pose_sums`, `synapses`), simulated and wall-clock time, the fit errors of
both calibrations, and `rows`.

One row is one held pose. It has every field of a row of the SO-3 repo (`error_deg` and so
on; see that repo's data README) and:

| Field | Meaning |
| --- | --- |
| `true_position` | exact position of the end point, base frame |
| `position`, `position_error_m` | position from the sheets' spikes of the last 0.5 s (each coordinate the mean of two sheets), and its distance from the true one; null if a coordinate has no spikes on either sheet |
| `cells_position`, `cells_error_m` | the same from the three position cells (mean of the last ten ticks) |
| `held_position_error_m` | exact position of the angles read back from the joint rings' activity, against the true position |
| `disagreement_m` | for x, y, z: the difference between the two sheets that hold the coordinate |
| `sheet_error_m` | for the sheets xy, xz, yz: distance of the packet from the true position in that plane |
| `sheet_spikes` | spikes of each sheet in the 0.5 s |

## A tracking, dropout or trail file

`ticks`: one entry per 50 ms.

| Field | Meaning |
| --- | --- |
| `t`, `q`, `moving`, `joint_speed` | time, joint angles, whether the arm is on its way, the fastest joint |
| `true_position`, `true_velocity` | of the end point, exact |
| `position`, `position_error_m` | from the sheets' spikes of the last two ticks |
| `position_tick`, `tick_error_m` | from this tick's spikes alone |
| `cells_position`, `cells_error_m` | from the position cells |
| `velocity` | from the velocity populations, m/s |
| `disagreement_m`, `sheet_spikes` | as above, for this tick |
| `error_deg`, `true_lift_pitch_yaw`, `ring_lift_pitch_yaw` | orientation, as in the SO-3 repo |
| `anchor`, `path_integration` | the two switches |
| `goal` | tracking: index of the goal the arm is moving to |
| `phase`, `gap`, `gap_s` | dropout and trail: `before`, `gap` (anchor off) or `after`; which gap; its length |

## `parts.json`

`hold`: one row per place (`drift_cells`: how far the packet moved in 10 s). `anchor`: one
row per place (`error_cells`). `speed`: one row per run (`value`: the two velocity values
handed in; `wanted` and `velocity`: asked and measured packet velocity, cells/s).
`velocity_cells`: per model, rows of `true` and `decoded` end-point velocity. `curve_*`:
the speed curve stored in the pose calibration.
