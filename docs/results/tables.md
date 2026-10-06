# Tables of the results

Written by `scripts/report.py` from `docs/results/data/`; do not edit by hand. Position error is the distance between the
decoded and the true end point, as a share of the arm's reach (the radius of its workspace, section 1) and in mm.
Median / p90 / max: half of the poses are at or below the median, 90% at or below p90. "Similar", fixed before the runs:
median at most 2% of the reach and p90 at most 4%. Reading and figures: [`../results.md`](../results.md).

## 0. The parts on their own (test 3)

Each new part measured in NEST on its own (`scripts/run_parts.py`).

| Part | Measured on | Result | Pass mark | Met |
| --- | --- | --- | --- | --- |
| a packet left alone | 8 places, 10 s each, no anchor, no velocity | alive in 8 of 8; moved 0.33 / 0.93 / 1.07 cells (median / p90 / largest), that is 1.65 / 4.67 / 5.33% of the reach | at most 2% of the reach in 10 s | NO |
| a sheet anchored through gate populations | 51 random places in the workspace, held 1 s each | packet off by 0.14 / 0.28 / 0.56 cells (median / p90 / largest), that is 0.71 / 1.40 / 2.82% of the reach; from one 50 ms tick alone 0.27 cells in the median | within 1 cell | yes |
| a free packet moved by a known velocity | 79 runs of 3 s: 8 directions, speeds from 15% to 90% of the top speed, two starts each (1 more left the sheet or fell silent) | speed along the asked direction over the asked speed: median 1.02, 10th to 90th percentile 0.66 to 1.71; velocity off by 33 / 104 / 153% of the asked speed (median / p90 / largest) | proportional to within 10% | NO |
| velocity populations of generic3 | 30 random poses, random joint velocities up to 0.5 rad/s handed in at rest | velocity of the end point off by 4 / 20 / 43% of the true speed (median / p90 / largest) | (none set) | |
| velocity populations of tiago7 | 30 random poses, random joint velocities up to 0.5 rad/s handed in at rest | velocity of the end point off by 5 / 13 / 19% of the true speed (median / p90 / largest) | (none set) | |

One cell is 0.050 of the reach. Calibration: a gate population gives its value to 0.0081 and its square to 0.0147 (root mean square); a bias cell fires at 99.01 Hz; the packet's top speed is a velocity value of 0.32, that is 0.47 of the reach per second.

The speed curve of the calibration (a sheet on its own):

| Drive on one side above the resting drive (share of the full drive) | 0.000 | 0.025 | 0.050 | 0.075 | 0.100 | 0.125 | 0.150 | 0.175 | 0.200 | 0.225 | 0.250 | 0.275 | 0.300 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Packet speed, mean of four directions (cells/s) | 0.00 | 0.74 | 1.64 | 2.25 | 3.57 | 4.35 | 5.33 | 6.95 | 7.25 | 7.81 | 8.18 | 8.68 | 9.47 |
| Slowest and fastest of the four directions | -0.0 - 0.0 | 0.6 - 1.0 | 1.6 - 1.8 | 1.6 - 3.2 | 2.5 - 4.3 | 3.5 - 5.1 | 4.7 - 6.1 | 6.5 - 7.6 | 6.9 - 7.8 | 6.6 - 8.6 | 6.9 - 9.0 | 7.2 - 9.4 | 8.4 - 11.0 |

## 1. The models

| Model | Joints | Reach (m) | Neurons of the SO(3) part | Products for the velocity | Sum populations | Neurons in all | Synapses |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| generic2 | 2 | 0.400 | 1,860 | 8 | 9 | 11,276 | 11,348,670 |
| generic3 | 3 | 0.600 | 4,420 | 17 | 12 | 15,876 | 12,576,110 |
| generic4 | 4 | 0.800 | 7,500 | 26 | 15 | 20,996 | 13,921,910 |
| generic5 | 5 | 1.000 | 10,580 | 35 | 18 | 26,116 | 15,097,910 |
| generic6 | 6 | 1.200 | 13,660 | 44 | 21 | 31,236 | 16,205,650 |
| generic7 | 7 | 1.400 | 16,740 | 53 | 24 | 36,356 | 17,378,190 |
| tiago2 | 2 | 0.220 | 1,860 | 8 | 9 | 11,276 | 11,367,070 |
| tiago3 | 3 | 0.444 | 4,420 | 17 | 12 | 15,876 | 12,709,310 |
| tiago4 | 4 | 0.610 | 7,500 | 26 | 15 | 20,996 | 14,053,910 |
| tiago5 | 5 | 0.760 | 10,580 | 35 | 18 | 26,116 | 15,203,510 |
| tiago6 | 6 | 0.760 | 13,660 | 35 | 18 | 29,256 | 15,844,450 |
| tiago7 | 7 | 0.806 | 16,740 | 53 | 24 | 36,356 | 17,472,990 |

## 2. Held poses: position (test 4)

The arm moves to the pose and rests 1 s; the position is the centre of the sheets' spikes of the last 0.5 s.

| Model | Poses | Position error, median / p90 / max (% of reach) | The same in mm | 95% interval of the median (%) | Within 4% of reach | Similar | Sheets silent |
| --- | ---: | --- | --- | --- | ---: | --- | ---: |
| generic2 | 300 | 0.98 / 1.62 / 2.58 | 3.9 / 6.5 / 10.3 | 0.91 - 1.05 | 100% | yes | 0 |
| generic3 | 300 | 1.03 / 1.70 / 2.61 | 6.2 / 10.2 / 15.6 | 0.94 - 1.08 | 100% | yes | 0 |
| generic4 | 300 | 0.97 / 1.65 / 2.99 | 7.8 / 13.2 / 24.0 | 0.89 - 1.04 | 100% | yes | 0 |
| generic5 | 300 | 0.99 / 1.79 / 3.26 | 9.9 / 17.9 / 32.6 | 0.93 - 1.06 | 100% | yes | 0 |
| generic6 | 300 | 1.00 / 1.83 / 2.95 | 12.0 / 21.9 / 35.4 | 0.94 - 1.09 | 100% | yes | 0 |
| generic7 | 300 | 1.01 / 1.85 / 3.71 | 14.1 / 25.9 / 52.0 | 0.93 - 1.06 | 100% | yes | 0 |
| tiago2 | 300 | 1.05 / 1.84 / 3.16 | 2.3 / 4.0 / 6.9 | 0.98 - 1.12 | 100% | yes | 0 |
| tiago3 | 300 | 1.11 / 1.89 / 2.99 | 4.9 / 8.4 / 13.3 | 1.03 - 1.17 | 100% | yes | 0 |
| tiago4 | 300 | 1.17 / 1.90 / 3.17 | 7.1 / 11.6 / 19.3 | 1.08 - 1.25 | 100% | yes | 0 |
| tiago5 | 300 | 1.35 / 2.23 / 4.48 | 10.3 / 17.0 / 34.0 | 1.30 - 1.42 | 99% | yes | 0 |
| tiago6 | 300 | 1.28 / 2.34 / 3.28 | 9.8 / 17.8 / 24.9 | 1.15 - 1.37 | 100% | yes | 0 |
| tiago7 | 300 | 1.37 / 2.30 / 4.07 | 11.1 / 18.6 / 32.8 | 1.30 - 1.47 | 100% | yes | 0 |

## 3. Where the position is read (test 5)

The same poses. *Position cells alone*: the three populations that hold the position before the sheets. *Exact position of the angles the joint rings hold*: what an exact calculation gives from the rings' own activity; no readout of those rings can do better.

| Model | Position cells alone, median / p90 (% of reach) | Sheets | Exact position of the angles the joint rings hold | Median of (sheets - cells) on the same poses | Sheets closer |
| --- | --- | --- | --- | ---: | ---: |
| generic2 | 0.73 / 1.41 | 0.98 / 1.62 | 0.14 / 0.33 | +0.17 | 31% |
| generic3 | 0.79 / 1.42 | 1.03 / 1.70 | 0.15 / 0.36 | +0.20 | 26% |
| generic4 | 0.78 / 1.43 | 0.97 / 1.65 | 0.15 / 0.35 | +0.16 | 30% |
| generic5 | 0.82 / 1.58 | 0.99 / 1.79 | 0.14 / 0.30 | +0.15 | 28% |
| generic6 | 0.86 / 1.68 | 1.00 / 1.83 | 0.15 / 0.31 | +0.13 | 31% |
| generic7 | 0.86 / 1.63 | 1.01 / 1.85 | 0.14 / 0.32 | +0.11 | 34% |
| tiago2 | 0.90 / 1.60 | 1.05 / 1.84 | 0.10 / 0.24 | +0.16 | 32% |
| tiago3 | 0.99 / 1.85 | 1.11 / 1.89 | 0.12 / 0.27 | +0.10 | 35% |
| tiago4 | 1.09 / 1.86 | 1.17 / 1.90 | 0.14 / 0.28 | +0.09 | 38% |
| tiago5 | 1.23 / 2.12 | 1.35 / 2.23 | 0.15 / 0.29 | +0.14 | 33% |
| tiago6 | 1.19 / 2.15 | 1.28 / 2.34 | 0.15 / 0.30 | +0.08 | 39% |
| tiago7 | 1.28 / 2.23 | 1.37 / 2.30 | 0.16 / 0.30 | +0.08 | 42% |

## 4. Agreement of the sheets (test 6)

Each coordinate is held by two sheets.

| Model | Difference of the two copies of a coordinate, median / p90 / max (% of reach) | Error of one sheet alone in its plane, median / p90 (% of reach): xy | xz | yz | Spikes of a sheet in the 0.5 s read, median |
| --- | --- | --- | --- | --- | ---: |
| generic2 | 0.34 / 0.93 / 2.05 | 0.77 / 1.54 | 0.78 / 1.41 | 0.90 / 1.65 | 223 |
| generic3 | 0.31 / 0.84 / 1.94 | 0.78 / 1.48 | 0.79 / 1.49 | 0.86 / 1.63 | 220 |
| generic4 | 0.31 / 0.81 / 1.77 | 0.71 / 1.48 | 0.82 / 1.53 | 0.82 / 1.60 | 224 |
| generic5 | 0.31 / 0.83 / 1.68 | 0.77 / 1.52 | 0.83 / 1.59 | 0.84 / 1.61 | 222 |
| generic6 | 0.32 / 0.80 / 2.57 | 0.82 / 1.56 | 0.81 / 1.74 | 0.89 / 1.67 | 225 |
| generic7 | 0.31 / 0.86 / 2.03 | 0.83 / 1.53 | 0.84 / 1.54 | 0.82 / 1.69 | 222 |
| tiago2 | 0.33 / 0.82 / 2.04 | 1.00 / 1.80 | 0.82 / 1.58 | 0.74 / 1.51 | 216 |
| tiago3 | 0.30 / 0.80 / 2.11 | 0.96 / 1.86 | 0.84 / 1.71 | 0.86 / 1.62 | 217 |
| tiago4 | 0.34 / 0.87 / 2.17 | 0.98 / 1.93 | 0.88 / 1.77 | 0.83 / 1.73 | 219 |
| tiago5 | 0.31 / 0.82 / 1.71 | 1.17 / 2.06 | 1.07 / 1.96 | 1.02 / 1.84 | 222 |
| tiago6 | 0.31 / 0.82 / 2.07 | 1.09 / 2.12 | 1.04 / 1.99 | 0.98 / 1.82 | 224 |
| tiago7 | 0.30 / 0.82 / 1.95 | 1.22 / 2.26 | 1.02 / 2.00 | 1.04 / 1.89 | 224 |

## 5. Orientation in the larger network (test 10)

| Model | Orientation error here, median / p90 / max (deg) | In the SO-3 repo, same poses | Within the SO(3) mark (median <= 5, p90 <= 10) |
| --- | --- | --- | --- |
| generic2 | 0.44 / 0.76 / 1.14 | 0.43 / 0.78 / 1.29 | yes |
| generic3 | 0.58 / 1.03 / 1.69 | 0.60 / 1.04 / 1.44 | yes |
| generic4 | 0.78 / 1.17 / 1.93 | 0.77 / 1.22 / 2.06 | yes |
| generic5 | 0.91 / 1.51 / 2.13 | 0.91 / 1.45 / 2.24 | yes |
| generic6 | 0.95 / 1.61 / 2.40 | 0.97 / 1.57 / 2.48 | yes |
| generic7 | 1.07 / 1.69 / 2.77 | 1.06 / 1.80 / 2.74 | yes |
| tiago2 | 0.44 / 0.76 / 1.07 | 0.44 / 0.75 / 1.26 | yes |
| tiago3 | 0.61 / 1.02 / 1.72 | 0.62 / 1.10 / 1.72 | yes |
| tiago4 | 0.80 / 1.30 / 1.85 | 0.80 / 1.29 / 1.96 | yes |
| tiago5 | 0.90 / 1.52 / 2.58 | 0.91 / 1.47 / 2.63 | yes |
| tiago6 | 0.97 / 1.61 / 2.38 | 1.03 / 1.66 / 2.48 | yes |
| tiago7 | 1.12 / 1.78 / 2.68 | 1.14 / 1.69 / 2.62 | yes |

## 6. A moving arm (test 7)

The arm goes from one random goal to the next and rests 1 s at each. The position is read from the sheets' spikes of the last two ticks. *Delay*: the shift in time for which the readout matches the true position best while the arm moves.

| Model | Joint speed limit (rad/s) | End point's speed while moving, median (reach/s) | Delay (s) | Position error while moving, median / p90 (% of reach) | At rest, median / p90 | 1 s after a stop, median / max | Held poses of the same model, p90 | Delay <= 0.5 s | Back at the held-pose level 1 s after a stop |
| --- | ---: | ---: | ---: | --- | --- | --- | ---: | --- | --- |
| generic3 | 0.25 | 0.17 | 0.20 | 3.78 / 5.87 | 1.38 / 2.24 | 1.56 / 2.56 | 1.70 | yes | yes |
| generic3 | 0.50 | 0.33 | 0.21 | 7.40 / 11.57 | 1.41 / 2.41 | 1.22 / 2.74 | 1.70 | yes | yes |
| generic3 | 1.00 | 0.67 | 0.21 | 14.14 / 22.92 | 1.41 / 2.29 | 1.49 / 2.84 | 1.70 | yes | yes |
| generic7 | 0.25 | 0.12 | 0.22 | 2.89 / 5.62 | 1.49 / 2.62 | 1.42 / 2.33 | 1.85 | yes | yes |
| generic7 | 0.50 | 0.22 | 0.23 | 5.25 / 11.72 | 1.38 / 2.40 | 1.42 / 2.48 | 1.85 | yes | yes |
| generic7 | 1.00 | 0.40 | 0.23 | 9.18 / 22.16 | 1.38 / 2.36 | 1.52 / 2.76 | 1.85 | yes | yes |
| tiago3 | 0.25 | 0.17 | 0.21 | 3.66 / 6.66 | 1.42 / 2.38 | 1.26 / 2.91 | 1.89 | yes | yes |
| tiago3 | 0.50 | 0.31 | 0.21 | 6.49 / 13.19 | 1.42 / 2.44 | 1.68 / 3.44 | 1.89 | yes | yes |
| tiago3 | 1.00 | 0.43 | 0.21 | 10.77 / 25.91 | 1.52 / 2.69 | 1.62 / 3.30 | 1.89 | yes | yes |
| tiago7 | 0.25 | 0.11 | 0.21 | 3.02 / 6.00 | 1.54 / 2.51 | 1.44 / 2.24 | 2.30 | yes | yes |
| tiago7 | 0.50 | 0.21 | 0.23 | 5.26 / 11.96 | 1.47 / 2.32 | 1.56 / 2.47 | 2.30 | yes | yes |
| tiago7 | 1.00 | 0.38 | 0.25 | 9.61 / 23.33 | 1.49 / 2.58 | 1.31 / 4.30 | 2.30 | yes | yes |

## 6b. A moving arm: variants at 0.5 rad/s

| Model | Variant | Delay (s) | Position error while moving, median / p90 (% of reach) | The same against the true position one delay earlier, median | At rest, median / p90 | Position cells alone while moving, median | Orientation error while moving, median (deg) |
| --- | --- | ---: | --- | ---: | --- | ---: | ---: |
| generic3 | as built (anchor and path integration) | 0.21 | 7.40 / 11.57 | 1.84 | 1.41 / 2.41 | 5.93 | 6.06 |
| generic3 | anchor only (velocity cells silent) | 0.22 | 7.64 / 12.00 | 1.86 | 1.44 / 2.42 | 5.93 | 6.06 |
| generic3 | anchor given the position 0.2 s ahead (position + velocity x 0.2 s) | 0.00 | 2.17 / 3.93 | 2.17 | 1.36 / 2.45 | 5.93 | 6.06 |
| generic3 | joint angles written ahead (the SO(3) option) | 0.06 | 2.62 / 4.51 | 1.81 | 1.38 / 2.28 | 1.65 | 1.02 |
| tiago7 | as built (anchor and path integration) | 0.23 | 5.26 / 11.96 | 2.09 | 1.47 / 2.32 | 4.34 | 6.40 |
| tiago7 | anchor only (velocity cells silent) | 0.24 | 5.38 / 12.19 | 2.02 | 1.44 / 2.30 | 4.34 | 6.40 |
| tiago7 | anchor given the position 0.2 s ahead (position + velocity x 0.2 s) | 0.03 | 2.28 / 5.42 | 2.26 | 1.48 / 2.25 | 4.34 | 6.40 |
| tiago7 | joint angles written ahead (the SO(3) option) | 0.05 | 2.31 / 5.38 | 2.09 | 1.58 / 2.42 | 1.89 | 1.74 |

## 6c. The velocity of the end point

Read from the three velocity populations in the sessions of section 6, on ticks with the end point faster than 2% of the reach per second.

| Model | Joint speed limit (rad/s) | Velocity read from the velocity populations, off by (% of the true speed), median / p90 |
| --- | ---: | --- |
| generic3 | 0.25 | 9 / 16 |
| generic3 | 0.50 | 15 / 45 |
| generic3 | 1.00 | 29 / 72 |
| generic7 | 0.25 | 18 / 41 |
| generic7 | 0.50 | 23 / 47 |
| generic7 | 1.00 | 41 / 78 |
| tiago3 | 0.25 | 10 / 25 |
| tiago3 | 0.50 | 17 / 57 |
| tiago3 | 1.00 | 28 / 67 |
| tiago7 | 0.25 | 15 / 39 |
| tiago7 | 0.50 | 19 / 55 |
| tiago7 | 1.00 | 33 / 74 |

## 7. Without the anchor, arm at rest (test 8a, 8c)

The arm rests at a random pose, the anchor is switched off for a while and on again (five times per length). *Packet moved*: the distance between the readout just before the gap and at its end.

| Model | Anchor off for (s) | Gaps | Packets alive at the end | Packet moved, median / max (% of reach) | Within 2% of reach | Level of the readout with the anchor on, p90 of single ticks (% of reach) | Back at that level after (s), median / max | Back within 0.5 s |
| --- | ---: | ---: | ---: | --- | ---: | ---: | --- | --- |
| generic3 | 2 | 5 | 4 | 3.06 / 4.77 | 0 of 5 | 2.77 | 0.25 / 0.75 | NO |
| generic3 | 5 | 5 | 5 | 4.68 / 7.30 | 0 of 5 | 2.77 | 0.05 / 0.40 | yes |
| generic3 | 10 | 5 | 5 | 4.54 / 6.88 | 1 of 5 | 2.77 | 0.05 / 0.30 | yes |
| tiago7 | 2 | 5 | 5 | 5.00 / 10.41 | 0 of 5 | 3.23 | 0.25 / 0.30 | yes |
| tiago7 | 5 | 5 | 5 | 1.98 / 7.85 | 3 of 5 | 3.23 | 0.05 / 0.20 | yes |
| tiago7 | 10 | 5 | 5 | 4.71 / 5.67 | 1 of 5 | 3.23 | 0.05 / 0.70 | NO |

## 8. Without the anchor, arm moving (test 8b, 8c)

The arm keeps moving from goal to goal while the anchor is off (five gaps per length). With path integration off the packets get nothing and should stay; *error over how far the end point moved* is then about 1, and below 1 is what path integration gains.

| Model | Arm | Anchor off for (s) | Path integration | Gaps | End point moved meanwhile, median (% of reach) | Error at the end of the gap, median / max (% of reach) | Error over how far the end point moved, median | Packets alive | Back at the level with the anchor on after (s), median / max |
| --- | --- | ---: | --- | ---: | ---: | --- | ---: | ---: | --- |
| generic3 | arm moving, 0.1 rad/s | 2 | on | 5 | 16.4 | 22.1 / 31.3 | 1.16 | 4 | 0.10 / 0.35 |
| generic3 | arm moving, 0.1 rad/s | 5 | on | 5 | 46.4 | 59.0 / 62.2 | - | 2 | 0.10 / 0.40 |
| generic3 | arm moving, 0.1 rad/s | 10 | on | 5 | 63.8 | 70.5 / 94.3 | 1.21 | 3 | 0.05 / 0.30 |
| generic3 | arm moving, 0.1 rad/s | 2 | off | 5 | 16.4 | 14.6 / 21.0 | 1.00 | 5 | 0.15 / 0.25 |
| generic3 | arm moving, 0.1 rad/s | 5 | off | 5 | 46.4 | 47.2 / 61.3 | - | 4 | 0.20 / 0.25 |
| generic3 | arm moving, 0.1 rad/s | 10 | off | 5 | 63.8 | 64.1 / 73.0 | 1.04 | 5 | 0.10 / 0.15 |
| generic3 | arm moving, 0.25 rad/s | 2 | on | 5 | 48.2 | 23.5 / 30.4 | 0.52 | 3 | 0.20 / 0.25 |
| generic3 | arm moving, 0.25 rad/s | 5 | on | 5 | 73.3 | 50.1 / 69.7 | 0.73 | 5 | 0.10 / 0.25 |
| generic3 | arm moving, 0.25 rad/s | 10 | on | 5 | 32.0 | 45.9 / 73.5 | 1.16 | 5 | 0.05 / 0.25 |
| generic3 | arm moving, 0.25 rad/s | 2 | off | 5 | 48.2 | 50.6 / 59.8 | 1.21 | 5 | 0.10 / 0.55 |
| generic3 | arm moving, 0.25 rad/s | 5 | off | 5 | 73.3 | 79.7 / 86.1 | 1.05 | 5 | 0.05 / 0.15 |
| generic3 | arm moving, 0.25 rad/s | 10 | off | 5 | 32.0 | 37.3 / 117.4 | 1.01 | 5 | 0.05 / 0.15 |
| generic3 | arm moving, 0.5 rad/s | 2 | on | 5 | 57.4 | 53.6 / 86.3 | 0.93 | 4 | 0.05 / 0.70 |
| generic3 | arm moving, 0.5 rad/s | 5 | on | 5 | 79.6 | 78.7 / 102.9 | 0.90 | 4 | 0.05 / 0.20 |
| generic3 | arm moving, 0.5 rad/s | 2 | off | 5 | 57.4 | 61.7 / 117.1 | 1.10 | 5 | 0.05 / 0.70 |
| generic3 | arm moving, 0.5 rad/s | 5 | off | 5 | 79.6 | 78.6 / 119.0 | 1.04 | 5 | 0.05 / 0.25 |
| tiago7 | arm moving, 0.1 rad/s | 2 | on | 5 | 4.9 | 23.9 / 28.4 | 3.34 | 4 | 0.10 / 0.40 |
| tiago7 | arm moving, 0.1 rad/s | 5 | on | 5 | 15.2 | 43.6 / 46.1 | - | 4 | 0.10 / 0.75 |
| tiago7 | arm moving, 0.1 rad/s | 10 | on | 5 | 47.9 | 71.9 / 98.6 | 1.69 | 4 | 0.85 / 1.70 |
| tiago7 | arm moving, 0.1 rad/s | 2 | off | 5 | 4.9 | 10.9 / 16.0 | 1.35 | 4 | 0.10 / 0.15 |
| tiago7 | arm moving, 0.1 rad/s | 5 | off | 5 | 15.2 | 13.8 / 65.2 | 0.96 | 3 | 0.10 / 2.70 |
| tiago7 | arm moving, 0.1 rad/s | 10 | off | 5 | 47.9 | 50.9 / 100.4 | 1.05 | 5 | 0.90 / 1.70 |
| tiago7 | arm moving, 0.25 rad/s | 2 | on | 5 | 22.3 | 18.8 / 30.8 | 0.89 | 1 | 0.05 / 0.10 |
| tiago7 | arm moving, 0.25 rad/s | 5 | on | 5 | 61.6 | 62.0 / 82.2 | 0.89 | 4 | 0.05 / 0.15 |
| tiago7 | arm moving, 0.25 rad/s | 10 | on | 5 | 71.5 | 104.8 / 158.9 | 1.56 | 4 | 0.05 / 0.15 |
| tiago7 | arm moving, 0.25 rad/s | 2 | off | 5 | 22.3 | 20.3 / 30.9 | 1.16 | 3 | 0.05 / 0.10 |
| tiago7 | arm moving, 0.25 rad/s | 5 | off | 5 | 61.6 | 69.3 / 118.1 | 1.05 | 5 | 0.05 / 0.15 |
| tiago7 | arm moving, 0.25 rad/s | 10 | off | 5 | 71.5 | 61.1 / 115.7 | 0.97 | 5 | 0.05 / 0.15 |
| tiago7 | arm moving, 0.5 rad/s | 2 | on | 5 | 14.4 | 13.2 / 73.5 | 0.98 | 3 | 0.05 / 1.45 |
| tiago7 | arm moving, 0.5 rad/s | 5 | on | 5 | 72.7 | 65.7 / 106.3 | 0.94 | 4 | 0.05 / 0.15 |
| tiago7 | arm moving, 0.5 rad/s | 2 | off | 5 | 14.4 | 16.0 / 79.5 | - | 3 | 0.05 / 1.45 |
| tiago7 | arm moving, 0.5 rad/s | 5 | off | 5 | 72.7 | 73.5 / 123.0 | 1.01 | 4 | 0.05 / 0.15 |

## 8b. The shift-register experiment

The arm moves at up to 0.25 rad/s; for 2 s the network gets neither the anchor nor the joint velocities. A shift register outside the network keeps the position read from the sheets in the last 10 ticks (0.5 s); the velocity a straight line through them shows is handed to the velocity cells when the gap starts.

| Model | Handed to the velocity cells during the gap | Gaps | End point moved meanwhile, median (% of reach) | Error at the end of the gap, median / max (% of reach) | Error over how far the end point moved, median |
| --- | --- | ---: | ---: | --- | ---: |
| generic3 | nothing (the packets get no input) | 10 | 30.8 | 33.5 / 51.1 | 1.17 |
| generic3 | the velocity the shift register shows, kept for the whole gap | 10 | 30.8 | 36.3 / 46.3 | 1.18 |
| generic3 | the joint velocities (path integration as in section 8) | 10 | 30.8 | 14.8 / 39.7 | 0.83 |
| tiago7 | nothing (the packets get no input) | 10 | 19.3 | 22.2 / 48.6 | 1.09 |
| tiago7 | the velocity the shift register shows, kept for the whole gap | 10 | 19.3 | 24.6 / 40.0 | 1.24 |
| tiago7 | the joint velocities (path integration as in section 8) | 10 | 19.3 | 21.3 / 28.9 | 1.09 |

## 9. Variants and robustness (test 9)

The first 100 random poses of the static test.

| Model | Variant | Poses | Position error, median / p90 / max (% of reach) | Median of (variant - as built) on the same poses | Sheets silent | Similar | Orientation error, median (deg) | Neurons |
| --- | --- | ---: | --- | ---: | ---: | --- | ---: | ---: |
| generic4 | as built | 100 | 0.94 / 1.64 / 2.19 | 0 | 0 | yes | 0.78 | 20,996 |
| generic4 | 5% of the sheets' neurons off | 100 | 1.10 / 1.93 / 2.55 | +0.17 | 0 | yes | 0.71 | 20,996 |
| generic4 | 10% of the sheets' neurons off | 100 | 1.34 / 2.00 / 3.08 | +0.33 | 0 | yes | 0.73 | 20,996 |
| generic4 | 20% of the sheets' neurons off | 100 | 1.58 / 2.37 / 3.42 | +0.51 | 0 | yes | 0.73 | 20,996 |
| generic4 | 5% of the SO(3) product neurons off | 100 | 4.42 / 8.16 / 14.47 | +3.52 | 0 | NO | 3.76 | 20,996 |
| generic4 | noise current 50 pA | 100 | 1.24 / 2.20 / 2.92 | +0.18 | 0 | yes | 0.73 | 20,996 |
| generic4 | noise current 150 pA | 100 | 3.69 / 6.07 / 8.38 | +2.77 | 0 | NO | 0.83 | 20,996 |
| generic4 | sheets of 32 x 32 | 100 | 1.12 / 1.92 / 2.53 | +0.14 | 0 | yes | 0.73 | 17,156 |
| generic4 | sheets of 64 x 64 | 100 | 0.86 / 1.48 / 2.43 | -0.09 | 1 | NO | 0.71 | 26,372 |
| generic4 | other neurons (seed 2) | 100 | 0.88 / 1.45 / 2.21 | -0.10 | 0 | yes | 0.73 | 20,996 |
| generic4 | other neurons (seed 3) | 100 | 1.01 / 1.63 / 2.57 | +0.10 | 0 | yes | 0.73 | 20,996 |
| generic4 | no velocity path (anchor only) | 100 | 0.85 / 1.48 / 1.83 | -0.11 | 0 | yes | 0.73 | 14,788 |
| tiago7 | as built | 100 | 1.32 / 2.21 / 4.07 | 0 | 0 | yes | 1.19 | 36,356 |
| tiago7 | 5% of the sheets' neurons off | 100 | 1.58 / 2.84 / 4.17 | +0.22 | 0 | yes | 1.19 | 36,356 |
| tiago7 | 10% of the sheets' neurons off | 100 | 1.61 / 3.20 / 4.35 | +0.26 | 0 | yes | 1.19 | 36,356 |
| tiago7 | 20% of the sheets' neurons off | 100 | 1.93 / 3.19 / 5.22 | +0.53 | 0 | yes | 1.19 | 36,356 |
| tiago7 | 5% of the SO(3) product neurons off | 100 | 7.21 / 11.58 / 15.73 | +5.96 | 0 | NO | 5.26 | 36,356 |
| tiago7 | noise current 50 pA | 100 | 1.91 / 3.40 / 4.82 | +0.58 | 0 | yes | 1.03 | 36,356 |
| tiago7 | noise current 150 pA | 100 | 6.31 / 8.48 / 11.05 | +4.80 | 0 | NO | 1.40 | 36,356 |
| tiago7 | sheets of 32 x 32 | 100 | 1.57 / 2.61 / 4.56 | +0.20 | 0 | yes | 1.19 | 32,516 |
| tiago7 | sheets of 64 x 64 | 100 | 1.27 / 2.18 / 3.87 | -0.06 | 1 | NO | 1.19 | 41,732 |
| tiago7 | other neurons (seed 2) | 100 | 1.31 / 2.23 / 2.98 | -0.04 | 0 | yes | 1.02 | 36,356 |
| tiago7 | other neurons (seed 3) | 100 | 1.50 / 2.83 / 3.58 | +0.15 | 0 | yes | 1.00 | 36,356 |
| tiago7 | no velocity path (anchor only) | 100 | 1.31 / 2.18 / 3.96 | -0.02 | 0 | yes | 1.19 | 24,028 |

## 10. Size and speed

| Model | Neurons | Of them the SO(3) part | Synapses | Simulated time per wall-clock time, 8 threads |
| --- | ---: | ---: | ---: | ---: |
| generic2 | 11,276 | 1,860 | 11.3 million | 2.48 |
| generic3 | 15,876 | 4,420 | 12.6 million | 1.50 |
| generic4 | 20,996 | 7,500 | 13.9 million | 1.02 |
| generic5 | 26,116 | 10,580 | 15.1 million | 0.74 |
| generic6 | 31,236 | 13,660 | 16.2 million | 0.59 |
| generic7 | 36,356 | 16,740 | 17.4 million | 0.45 |
| tiago2 | 11,276 | 1,860 | 11.4 million | 2.29 |
| tiago3 | 15,876 | 4,420 | 12.7 million | 1.44 |
| tiago4 | 20,996 | 7,500 | 14.1 million | 0.98 |
| tiago5 | 26,116 | 10,580 | 15.2 million | 0.72 |
| tiago6 | 29,256 | 13,660 | 15.8 million | 0.61 |
| tiago7 | 36,356 | 16,740 | 17.5 million | 0.44 |

Machine: AMD Ryzen 9 7900X 12-Core Processor, 24 hardware threads; nothing else was running.

## 11. Copy without spikes (test 2)

The position cells only; the sheets have no copy without spikes.

| Model | Position cells in the copy without spikes, median / p90 (% of reach) | Position cells in the spiking network |
| --- | --- | --- |
| generic2 | 1.03 / 2.81 | 0.73 / 1.41 |
| generic3 | 1.26 / 2.77 | 0.79 / 1.42 |
| generic4 | 1.56 / 2.79 | 0.78 / 1.43 |
| generic5 | 1.67 / 3.32 | 0.82 / 1.58 |
| generic6 | 2.31 / 4.16 | 0.86 / 1.68 |
| generic7 | 2.17 / 4.01 | 0.86 / 1.63 |
| tiago2 | 1.03 / 2.11 | 0.90 / 1.60 |
| tiago3 | 1.48 / 3.57 | 0.99 / 1.85 |
| tiago4 | 2.17 / 3.86 | 1.09 / 1.86 |
| tiago5 | 2.63 / 4.72 | 1.23 / 2.12 |
| tiago6 | 2.69 / 4.54 | 1.19 / 2.15 |
| tiago7 | 2.81 / 4.75 | 1.28 / 2.23 |

## 12. The pass marks

Set before the tests were run (docs/tests.md). Tests without a mark (2, 5, 6, 8b, 9) are reported in their sections.

| Test | Pass mark | Outcome | Met |
| --- | --- | --- | --- |
| 1. Maths | position equal to 1e-9; velocity to 1e-6 | asserted by `tests/test_kinematics.py` | yes |
| 3. Parts: anchored packet | within 1 cell | largest 0.56 cells | yes |
| 3. Parts: packet speed | proportional to the velocity, to within 10% | off by 33% in the median, 104% at the 90th percentile | NO |
| 4. Held poses: position | median at most 2% of the reach and p90 at most 4%, every model | 12 of 12 models; medians 0.97% to 1.37%, largest p90 2.34% | yes |
| 10. Orientation unchanged | within the SO(3) marks (median at most 5 deg, p90 at most 10 deg) | 12 of 12 models; largest median 1.12 deg | yes |
| 7. Moving arm: delay | at most 0.5 s at 0.5 rad/s | 0.21 to 0.23 s (4 models) | yes |
| 7. Moving arm: after a stop | back at the held-pose level 1 s after a stop | 12 of 12 sessions | yes |
| 8a. Packet at rest without anchor | moves at most 2% of the reach in 10 s | median 4.59%, largest 6.88% (10 gaps); all alive: yes | NO |
| 8c. Anchor back on | readout back at its level with the anchor on within 0.5 s | median 0.10 s, largest 2.70 s; later than 0.5 s or not within the 3 s logged: 15 of 190 gaps | NO |
