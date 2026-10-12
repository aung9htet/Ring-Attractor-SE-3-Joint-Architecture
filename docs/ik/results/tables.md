# Inverse kinematics for a pose: tables

Written by `scripts/ik_report.py` from `docs/ik/results/data/`. Do not edit by hand.

Everything here is the ring IK, the spiking network of this repo, unless a row names another method. `tiago4`: TIAGo arm, 4 joints; `tiago7`: TIAGo arm, 7 joints; `generic4`: generic arm, 4 joints; `generic7`: generic arm, 7 joints. These are the arms the network was built for, not names of the method.

## 0. The marks, set before the test pairs were run

| Test | Mark | Result | |
| --- | --- | --- | --- |
| 3a | direction of the turn within 10° (median) | 4.6° on the worse of 2 runs | met |
| 3a | the offset within 2% of the reach (median) | 1.74% on the worst of 3 runs | met |
| 3b | right sign in 95% of the shares above 0.05 rad | 100% on the worst of 3 runs | met |
| 4 | tiago4, position: 85% of the goals reached | 98% of 40 | met |
| 4 | tiago4: 90% of the goals the exact rule reaches | 100% | met |
| 4 | tiago4: final error of reached goals 3% of the reach (median) | 1.95% | met |
| 5 | tiago4: path at most 1.5 times the straight line (median) | 1.36 | met |
| 5 | tiago4: within the marks after at most 10 s (median) | 4.5 s | met |
| 4 | generic4, position: 85% of the goals reached | 100% of 40 | met |
| 4 | generic4: 90% of the goals the exact rule reaches | 100% | met |
| 4 | generic4: final error of reached goals 3% of the reach (median) | 1.24% | met |
| 5 | generic4: path at most 1.5 times the straight line (median) | 1.32 | met |
| 5 | generic4: within the marks after at most 10 s (median) | 4.0 s | met |
| 4 | generic7, pose: 85% of the goals reached | 97% of 30 | met |
| 4 | generic7: 90% of the goals the exact rule reaches | 97% | met |
| 4 | generic7: final error of reached goals 3° and 3% of the reach (medians) | 2.27°, 1.97% | met |
| 5 | generic7: path at most 1.5 times the straight line (median) | 1.54 | **missed** |
| 5 | generic7: within the marks after at most 10 s (median) | 5.0 s | met |
| 4 | tiago7, pose: 85% of the goals reached | 93% of 30 | met |
| 4 | tiago7: 90% of the goals the exact rule reaches | 97% | met |
| 4 | tiago7: final error of reached goals 3° and 3% of the reach (medians) | 1.95°, 2.02% | met |
| 5 | tiago7: path at most 1.5 times the straight line (median) | 1.43 | met |
| 5 | tiago7: within the marks after at most 10 s (median) | 5.7 s | met |
| 10a | Gazebo's end point within 1 mm of the chain's kinematics | 0.22 mm at most, tiago7 at 5 poses, tiago4 at 5 poses | met |
| 10b | tiago7 in Gazebo, ring IK: 80% of the goals it reaches on the light robot | 28 of 28 (100%) | met |

## 1. The rule itself (exact kinematics, no neurons)

60 start and goal pairs per arm (seed 20261076, not the test pairs); a goal is the pose of a joint pose within 0.8 rad of the start, joint by joint, unless a row says otherwise. Delay 0.4 s, share noise 0.01 rad, decoded pose off by 1° and 1.3% of the reach. Balance 3, loop gain 1 per second (all joints at once and Jacobian transpose: 0.5). tiago4 and generic4 are asked for the position alone, generic7 and tiago7 for the pose. Reached: within 5° and 4% of the reach at the end of 25 s. The path is that of the whole 25 s, with the hand's wandering under the share noise after it has arrived.

What a share is divided by, the cascade, and two standard methods:

| Arm | Rule | Reached | Final position error, median | Final orientation error, median | Within the marks after | Path / straight | Missed (at a joint limit) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| tiago4 | cascade (as built: divided by the mean) | 100% | 1.41% | – | 4.5 s | 1.39 | 0 (0) |
| tiago4 | cascade, divided by the largest value | 93% | 1.60% | – | 5.6 s | 1.37 | 4 (2) |
| tiago4 | cascade, divided by the exact value | 100% | 1.44% | – | 4.2 s | 1.36 | 0 (0) |
| tiago4 | all joints at once | 92% | 2.11% | – | 8.2 s | 1.20 | 5 (1) |
| tiago4 | damped least squares | 100% | 1.31% | – | 4.2 s | 1.48 | 0 (0) |
| tiago4 | Jacobian transpose | 98% | 1.42% | – | 6.5 s | 1.44 | 1 (1) |
| generic4 | cascade (as built: divided by the mean) | 98% | 1.34% | – | 4.1 s | 1.41 | 1 (0) |
| generic4 | cascade, divided by the largest value | 93% | 1.44% | – | 6.7 s | 1.35 | 4 (0) |
| generic4 | cascade, divided by the exact value | 100% | 1.34% | – | 4.0 s | 1.40 | 0 (0) |
| generic4 | all joints at once | 98% | 1.50% | – | 7.8 s | 1.21 | 1 (0) |
| generic4 | damped least squares | 100% | 1.31% | – | 2.6 s | 1.50 | 0 (0) |
| generic4 | Jacobian transpose | 98% | 1.36% | – | 5.3 s | 1.39 | 1 (0) |
| generic7 | cascade (as built: divided by the mean) | 95% | 1.47% | 1.08° | 4.9 s | 1.55 | 3 (0) |
| generic7 | cascade, divided by the largest value | 78% | 2.37% | 2.22° | 8.3 s | 1.35 | 13 (0) |
| generic7 | cascade, divided by the exact value | 98% | 1.40% | 1.09° | 5.0 s | 1.50 | 1 (0) |
| generic7 | all joints at once | 85% | 2.13% | 2.02° | 11.7 s | 1.32 | 9 (0) |
| generic7 | damped least squares | 100% | 1.30% | 1.04° | 4.2 s | 1.50 | 0 (0) |
| generic7 | Jacobian transpose | 85% | 1.70% | 1.58° | 8.9 s | 1.53 | 9 (0) |
| tiago7 | cascade (as built: divided by the mean) | 95% | 1.53% | 1.21° | 5.5 s | 1.47 | 3 (2) |
| tiago7 | cascade, divided by the largest value | 90% | 1.63% | 1.28° | 6.3 s | 1.44 | 6 (4) |
| tiago7 | cascade, divided by the exact value | 95% | 1.48% | 1.22° | 5.0 s | 1.49 | 3 (3) |
| tiago7 | all joints at once | 78% | 2.17% | 1.88° | 10.2 s | 1.23 | 13 (3) |
| tiago7 | damped least squares | 100% | 1.29% | 1.06° | 5.2 s | 1.49 | 0 (0) |
| tiago7 | Jacobian transpose | 87% | 2.12% | 1.86° | 10.9 s | 2.53 | 8 (4) |

The order of the cascade, and a second sweep:

| Arm | Rule | Reached | Final position error, median | Final orientation error, median | Within the marks after | Path / straight | Missed (at a joint limit) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| tiago4 | cascade, from the hand to the base | 98% | 1.43% | – | 4.3 s | 1.52 | 1 (0) |
| tiago4 | cascade, down the arm twice per tick | 100% | 1.37% | – | 3.8 s | 1.51 | 0 (0) |
| generic4 | cascade, from the hand to the base | 100% | 1.38% | – | 4.2 s | 1.51 | 0 (0) |
| generic4 | cascade, down the arm twice per tick | 97% | 1.32% | – | 3.0 s | 1.48 | 2 (0) |
| generic7 | cascade, from the hand to the base | 97% | 1.31% | 1.20° | 5.6 s | 1.57 | 2 (0) |
| generic7 | cascade, down the arm twice per tick | 97% | 1.33% | 1.06° | 3.5 s | 1.59 | 2 (0) |
| tiago7 | cascade, from the hand to the base | 97% | 1.43% | 1.23° | 5.7 s | 1.58 | 2 (1) |
| tiago7 | cascade, down the arm twice per tick | 98% | 1.32% | 1.16° | 4.3 s | 1.59 | 1 (1) |

The balance (how many radians of turn one reach of offset counts as):

| Arm | Rule | Reached | Final position error, median | Final orientation error, median | Within the marks after | Path / straight | Missed (at a joint limit) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| tiago4 | cascade, balance 2 | 97% | 1.79% | – | 5.7 s | 1.37 | 2 (0) |
| tiago4 | cascade, balance 4 | 100% | 1.37% | – | 4.2 s | 1.37 | 0 (0) |
| tiago4 | cascade, balance 5 | 100% | 1.35% | – | 3.8 s | 1.36 | 0 (0) |
| generic4 | cascade, balance 2 | 97% | 1.42% | – | 6.2 s | 1.40 | 2 (0) |
| generic4 | cascade, balance 4 | 98% | 1.34% | – | 3.4 s | 1.42 | 1 (0) |
| generic4 | cascade, balance 5 | 98% | 1.34% | – | 3.1 s | 1.41 | 1 (0) |
| generic7 | cascade, balance 2 | 95% | 1.58% | 1.03° | 7.6 s | 1.58 | 3 (0) |
| generic7 | cascade, balance 4 | 95% | 1.38% | 1.20° | 4.5 s | 1.51 | 3 (0) |
| generic7 | cascade, balance 5 | 95% | 1.34% | 1.38° | 5.2 s | 1.49 | 3 (0) |
| tiago7 | cascade, balance 2 | 92% | 1.68% | 1.14° | 6.2 s | 1.49 | 5 (2) |
| tiago7 | cascade, balance 4 | 93% | 1.50% | 1.26° | 5.2 s | 1.45 | 4 (3) |
| tiago7 | cascade, balance 5 | 90% | 1.41% | 1.35° | 5.2 s | 1.44 | 6 (5) |

How far the goal is:

| Arm | Rule | Reached | Final position error, median | Final orientation error, median | Within the marks after | Path / straight | Missed (at a joint limit) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| tiago4 | cascade, goals within 0.4 rad | 100% | 1.33% | – | 2.1 s | 1.74 | 0 (0) |
| tiago4 | cascade, goals within 1.2 rad | 92% | 1.62% | – | 6.1 s | 1.28 | 5 (1) |
| generic4 | cascade, goals within 0.4 rad | 100% | 1.33% | – | 1.8 s | 1.76 | 0 (0) |
| generic4 | cascade, goals within 1.2 rad | 97% | 1.35% | – | 5.6 s | 1.32 | 2 (0) |
| generic7 | cascade, goals within 0.4 rad | 100% | 1.19% | 1.00° | 2.1 s | 1.85 | 0 (0) |
| generic7 | cascade, goals within 1.2 rad | 88% | 1.61% | 1.45° | 8.8 s | 1.40 | 7 (0) |
| tiago7 | cascade, goals within 0.4 rad | 100% | 1.35% | 1.03° | 2.6 s | 1.83 | 0 (0) |
| tiago7 | cascade, goals within 1.2 rad | 88% | 1.65% | 1.67° | 8.3 s | 1.36 | 7 (4) |

Goals anywhere in the joint ranges:

| Arm | Rule | Reached | Final position error, median | Final orientation error, median | Within the marks after | Path / straight | Missed (at a joint limit) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| tiago4 | cascade, goals anywhere | 77% | 2.03% | – | 10.2 s | 1.25 | 14 (12) |
| tiago4 | cascade divided by the exact value, goals anywhere | 80% | 1.92% | – | 10.2 s | 1.23 | 12 (8) |
| tiago4 | damped least squares, goals anywhere | 75% | 1.36% | – | 11.9 s | 1.21 | 15 (12) |
| generic4 | cascade, goals anywhere | 65% | 2.31% | – | 11.9 s | 1.31 | 21 (0) |
| generic4 | cascade divided by the exact value, goals anywhere | 68% | 2.13% | – | 11.9 s | 1.28 | 19 (0) |
| generic4 | damped least squares, goals anywhere | 80% | 1.32% | – | 11.2 s | 1.18 | 12 (0) |
| generic7 | cascade, goals anywhere | 37% | 5.95% | 4.92° | 15.9 s | 1.49 | 38 (0) |
| generic7 | cascade divided by the exact value, goals anywhere | 45% | 4.49% | 3.74° | 18.1 s | 1.50 | 33 (0) |
| generic7 | damped least squares, goals anywhere | 77% | 1.31% | 1.07° | 15.4 s | 1.15 | 14 (0) |
| tiago7 | cascade, goals anywhere | 45% | 4.43% | 3.36° | 11.4 s | 1.34 | 33 (26) |
| tiago7 | cascade divided by the exact value, goals anywhere | 45% | 4.29% | 3.23° | 12.3 s | 1.26 | 33 (25) |
| tiago7 | damped least squares, goals anywhere | 53% | 1.88% | 1.81° | 14.9 s | 1.17 | 28 (27) |

What the rule needs (7 joints):

| Arm | Rule | Reached | Final position error, median | Final orientation error, median | Within the marks after | Path / straight | Missed (at a joint limit) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| generic7 | cascade, gain 0.75 | 95% | 1.57% | 1.10° | 6.3 s | 1.38 | 3 (0) |
| generic7 | cascade, gain 1.50 | 98% | 1.38% | 1.01° | 3.5 s | 1.95 | 1 (0) |
| generic7 | cascade, delay 0.25 s | 95% | 1.47% | 1.07° | 5.0 s | 1.49 | 3 (0) |
| generic7 | cascade, delay 0.55 s | 97% | 1.46% | 1.08° | 4.7 s | 1.65 | 2 (0) |
| generic7 | cascade, share noise 0.02 rad | 95% | 1.49% | 1.19° | 4.8 s | 1.85 | 3 (0) |
| generic7 | cascade, readout off by 2 deg and 2.6% | 95% | 2.64% | 2.04° | 6.0 s | 1.54 | 3 (0) |
| generic7 | cascade, nothing in the way | 93% | 0.77% | 0.71° | 5.0 s | 1.15 | 4 (0) |
| generic7 | orientation alone (the cascade of the SO-3 repo) | 100% | – | 0.99° | 2.5 s | – | 0 (0) |
| tiago7 | cascade, gain 0.75 | 93% | 1.55% | 1.21° | 6.8 s | 1.33 | 4 (3) |
| tiago7 | cascade, gain 1.50 | 97% | 1.48% | 1.19° | 4.0 s | 1.80 | 2 (2) |
| tiago7 | cascade, delay 0.25 s | 93% | 1.52% | 1.21° | 5.5 s | 1.44 | 4 (3) |
| tiago7 | cascade, delay 0.55 s | 95% | 1.52% | 1.22° | 5.7 s | 1.54 | 3 (2) |
| tiago7 | cascade, share noise 0.02 rad | 93% | 1.66% | 1.29° | 5.4 s | 1.86 | 4 (3) |
| tiago7 | cascade, readout off by 2 deg and 2.6% | 93% | 2.63% | 2.12° | 5.6 s | 1.47 | 4 (3) |
| tiago7 | cascade, nothing in the way | 95% | 0.73% | 0.67° | 5.8 s | 1.08 | 3 (3) |
| tiago7 | orientation alone (the cascade of the SO-3 repo) | 100% | – | 0.99° | 2.4 s | – | 0 (0) |

## 2. The sheets as the comparison needs them: level cells and the ramp readout (test 3c)

Three sheets on their own, anchored through gate populations at 80 random places of the workspace (240 packets; none of the places the readout was fitted on). Level cells: set point 500 spikes per second, gain 200 pA, synapse 100 ms. A packet counts as out below 100 spikes per second, as weak below 300.

| Sheets | Spikes per second of a packet, median | sd | smallest | largest | Packets out | Packets weak | Packet off its anchor, rms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| without level cells (as in the forward network) | 459 | 120 | 0 | 678 | 2 of 240 | 33 of 240 | 0.54% |
| with level cells (as in the IK network) | 494 | 24 | 418 | 548 | 0 of 240 | 0 of 240 | 0.52% |

The ramp readout (160 neurons per coordinate, fitted on sheets with level cells): the place it gives against the centre of the packets' spikes, % of the reach.

| Sheets | Coordinates read | rms | median | 90th percentile | largest | mean (bias) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| without level cells | 240 | 6.26% | 0.68% | 8.85% | 46.53% | 0.15% |
| with level cells | 240 | 0.59% | 0.32% | 0.98% | 2.09% | 0.16% |

Other level cells, on the first 30 of the places (90 packets). "From tick to tick": the standard deviation of a packet's spikes per second over the ten 50 ms ticks of a measurement, median over the packets.

| Level cells: gain, synapse from the sheet | Spikes per second of a packet, median | sd over the places | sd from tick to tick | Packets weak | Ramp readout, rms |
| --- | ---: | ---: | ---: | ---: | ---: |
| none | 480 | 112 | 61 | 9 of 90 | 3.95% |
| 200 pA, 100 ms (as built) | 496 | 22 | 69 | 0 of 90 | 0.54% |
| 100 pA, 30 ms | 494 | 35 | 88 | 0 of 90 | 0.51% |
| 300 pA, 30 ms | 586 | 52 | 632 | 0 of 90 | 2.04% |
| 100 pA, 100 ms | 492 | 36 | 67 | 0 of 90 | 0.66% |
| 300 pA, 150 ms | 496 | 18 | 76 | 0 of 90 | 0.52% |

From `docs/ik/spikes/figures.json` (three sheets with level cells, `scripts/ik_spike_census.py`): read from the spikes of one 50 ms tick, a coordinate scatters by 2.7% of the reach from tick to tick; read over ten ticks it is 1.1% from the place. While the anchored place travels at 0.3 reaches per second the packets are 0.06 s behind it and the ramp readouts 0.11 s.

In the calibration itself (771 measurements on a grid, each also with -25 and +25 pA added to every sheet neuron): fit 0.42% (rms); on 120 measurements between the grid's places 0.51% (rms), largest 1.76%.

## 3. Comparisons and shares in spikes, arm held by hand (tests 3a, 3b)

The arm is held at a pose; the goal is part of the way to a pose within 0.8 rad per joint. Marks: direction of the turn within 10° (median); the offset within 2% of the reach (median); right sign in 95% of the shares above 0.05 rad.

| Arm | Task | Poses | Neurons | Turn: direction error, median | p90 | Offset: error, median (% of reach) | direction error, median | Shares above 0.05 rad | Right sign | Share error, rms | Share where none is asked, rms |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| tiago4 | position | 60 | 35,916 | – | – | 1.36% | 7.8° | 80 | 100% | 0.0121 rad | 0.0112 rad |
| generic7 | pose | 40 | 71,976 | 2.9° | 15.0° | 1.37% | 9.8° | 148 | 100% | 0.0162 rad | 0.0140 rad |
| tiago7 | pose | 40 | 69,036 | 4.6° | 15.2° | 1.74% | 9.6° | 131 | 100% | 0.0169 rad | 0.0136 rad |

Where a position is read, in the same runs: error against the true position of the hand (or against the goal that was written), median, % of the reach.

| Arm | Position cells | Forward sheets (centre of the spikes) | Ramp readout of the forward sheets | Goal sheets (centre of the spikes) | Ramp readout of the goal sheets | Fewest spikes of a forward sheet per tick |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| tiago4 | 1.10% | 1.06% | 1.16% | 0.68% | 1.02% | 21.2 |
| generic7 | 0.86% | 0.95% | 1.13% | 0.71% | 1.00% | 20.6 |
| tiago7 | 1.22% | 1.31% | 1.39% | 0.82% | 1.21% | 21.0 |

Share error along the arm (rms, rad): does the cascade get worse from joint to joint?

- tiago4: joint 1 0.010, joint 2 0.011, joint 3 0.014, joint 4 0.013
- generic7: joint 1 0.017, joint 2 0.017, joint 3 0.017, joint 4 0.018, joint 5 0.015, joint 6 0.013, joint 7 0.016
- tiago7: joint 1 0.011, joint 2 0.012, joint 3 0.021, joint 4 0.017, joint 5 0.019, joint 6 0.017, joint 7 0.018

## 4. Reaching and trajectory in closed loop (tests 4, 5)

Spiking network, light robot, start and goal pairs of the test seed; a goal is the pose of a joint pose within 0.8 rad of the start, joint by joint. Reached: at the end of the trial the true position is within 4% of the reach of the goal and, where the task has an orientation, the true orientation within 5°.

| Run | Trials | Reached | Exact rule reaches | Reached, of those the exact rule reaches | Final position error, median / p90 | In mm, median | Final orientation error, median / p90 | Within the marks after, median | Path / straight, median | Missed (with a joint at a limit) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| tiago4, position | 40 | 98% | 95% | 100% | 1.96% / 3.16% | 12.0 | – | 4.5 s | 1.36 | 1 (0) |
| generic4, position | 40 | 100% | 100% | 100% | 1.24% / 2.03% | 9.9 | – | 4.0 s | 1.32 | 0 (0) |
| generic7, pose | 30 | 97% | 100% | 97% | 2.02% / 3.30% | 28.2 | 2.27° / 3.54° | 5.0 s | 1.54 | 1 (0) |
| tiago7, pose | 30 | 93% | 97% | 97% | 2.09% / 3.48% | 16.8 | 1.89° / 3.00° | 5.7 s | 1.43 | 2 (0) |

| Arm | Neurons | Of them the IK layer | Synapses | Simulated time per wall-clock time (12 threads, nothing else on the machine) | Built in | Trial length, median | Start: from the goal, median | Position read from the forward sheets off by, median | Decoded orientation off by, median | Exact rule: final error, median | Exact rule: within the marks after |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| tiago4 | 35,916 | 20,948 | 31,297,304 | 0.66 | 4.4 s | 25.0 s | 39% | 1.58% | 1.04° | 0.99% | 4.2 s |
| generic4 | 35,916 | 20,948 | 31,069,368 | 0.63 | 4.3 s | 15.5 s | 33% | 1.26% | 0.99° | 0.42% | 4.2 s |
| generic7 | 71,976 | 47,768 | 38,735,788 | 0.21 | 5.4 s | 15.1 s | 22%, 55° | 1.41% | 1.61° | 0.81%, 0.92° | 5.6 s |
| tiago7 | 69,036 | 44,828 | 38,472,696 | 0.22 | 5.3 s | 13.1 s | 31%, 55° | 1.68% | 1.46° | 0.80%, 0.45° | 5.2 s |

## 5. Variants on the same pairs (tests 6, 7)

| Arm, task | Variant | Trials | Reached | Exact rule reaches | Reached, of those the exact rule reaches | Final position error, median / p90 | In mm, median | Final orientation error, median / p90 | Within the marks after, median | Path / straight, median | Missed (with a joint at a limit) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| tiago4, position | as built: positions compared on the sheets | 40 | 98% | 95% | 100% | 1.96% / 3.16% | 12.0 | – | 4.5 s | 1.36 | 1 (0) |
| tiago4, position | positions compared as numbers (no sheet in the loop) | 40 | 98% | 95% | 100% | 1.62% / 2.84% | 9.9 | – | 4.8 s | 1.22 | 1 (0) |
| tiago4, position | every joint gets the whole error (no cascade), gain 0.5 | 40 | 88% | 88% | 100% | 2.33% / 4.90% | 14.2 | – | 8.1 s | 1.13 | 5 (1) |
| tiago7, pose | as built: positions compared on the sheets | 30 | 93% | 97% | 97% | 2.09% / 3.48% | 16.8 | 1.89° / 3.00° | 5.7 s | 1.43 | 2 (0) |
| tiago7, pose | positions compared as numbers (no sheet in the loop) | 20 | 95% | 100% | 95% | 1.70% / 3.19% | 13.7 | 1.63° / 3.12° | 5.8 s | 1.44 | 1 (0) |
| tiago7, orientation alone | the cascade of the SO-3 repo in this network | 20 | 100% | 100% | 100% | – | – | 1.54° / 2.25° | 3.1 s | – | 0 (0) |

- tiago4, position: sheets against numbers, the same 40 pairs: the first is closer to the goal at the end in 40% of them (median difference 0.14% of the reach).
- tiago7, pose: sheets against numbers, the same 20 pairs: the first is closer to the goal at the end in 30% of them (median difference 0.18% of the reach).
- tiago4, position: cascade against all joints at once, the same 40 pairs: the first is closer to the goal at the end in 65% of them (median difference -0.37% of the reach).

## 6. Disturbances (test 8)

tiago4, position goals, the first 20 test pairs.

| Condition | Trials | Reached | Exact rule reaches | Reached, of those the exact rule reaches | Final position error, median / p90 | In mm, median | Final orientation error, median / p90 | Within the marks after, median | Path / straight, median | Missed (with a joint at a limit) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| nothing in the way | 20 | 100% | 95% | 100% | 1.62% / 3.15% | 9.9 | – | 4.7 s | 1.40 | 0 (0) |
| the robot moves 30% less than told | 20 | 95% | 95% | 100% | 1.69% / 2.77% | 10.3 | – | 6.5 s | 1.21 | 1 (0) |
| the robot moves 30% more than told | 20 | 100% | 100% | 100% | 1.42% / 2.86% | 8.6 | – | 4.1 s | 1.50 | 0 (0) |
| one joint pushed 0.3 rad after 4 s | 20 | 100% | 100% | 100% | 1.52% / 2.87% | 9.3 | – | 5.7 s | 1.79 | 0 (0) |
| the goal changes after 4 s | 20 | 80% | 75% | 100% | 2.30% / 23.96% | 14.0 | – | 17.5 s | 3.25 | 4 (3) |
| goals anywhere in the joint ranges | 20 | 70% | 80% | 88% | 2.15% / 7.82% | 13.1 | – | 11.4 s | 1.22 | 6 (1) |

Of the goals reached with nothing in the way, reached under the disturbance: the robot moves 30% less than told: 19 of 20; the robot moves 30% more than told: 20 of 20; one joint pushed 0.3 rad after 4 s: 20 of 20; the goal changes after 4 s: 16 of 20.

## 7. How the trials ended (test 9)

| Arm | Ended | Trials | Reached | Exact rule reaches | Final position error, median | Final orientation error, median |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| tiago4 | with a joint at a limit | 3 | 100% | 100% | 3.15% | – |
| tiago4 | with no joint at a limit | 37 | 97% | 95% | 1.77% | – |
| generic4 | with no joint at a limit | 40 | 100% | 100% | 1.24% | – |
| generic7 | with no joint at a limit | 30 | 97% | 100% | 2.02% | 2.27° |
| tiago7 | with a joint at a limit | 2 | 100% | 100% | 1.75% | 3.17° |
| tiago7 | with no joint at a limit | 28 | 93% | 96% | 2.09% | 1.84° |

The goals that were missed:

- tiago4, pair 29: ended 7.0% of the reach from the goal; the exact rule ended 5.5% from it.
- generic7, pair 5: ended 4.7% of the reach, 8.0° from the goal; the exact rule ended 1.0%, 2.0° from it.
- tiago7, pair 4: ended 5.4% of the reach, 0.5° from the goal; the exact rule ended 3.5%, 0.7° from it.
- tiago7, pair 25: ended 4.2% of the reach, 1.1° from the goal; the exact rule ended 3.1%, 6.5° from it.

## 8. What the layer is made of

Counted from the wiring the network is built from. A product population has 200 neurons, a sum or a share half 60, a ring 200, a sheet 2304, a ramp readout 160.

| Arm, task | Goal orientation | Turn | Offset | Levers | Per joint: direction, share, what is left (products + sums or halves) | Products in all | Sums | Halves | Neurons: pose network | IK layer | Together |
| --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| tiago4, position | – | – | 0 + 3 | 0 + 9 | 1: 0 + 2, 2 + 2, 2 + 2; 2: 4 + 3, 3 + 2, 3 + 3; 3: 6 + 3, 3 + 2, 3 + 3; 4: 6 + 3, 3 + 2, 3 + 3 | 38 | 34 | 8 | 14,968 | 20,948 | 35,916 |
| tiago4, pose | 14 + 4 | 18 + 3 | 0 + 3 | 0 + 9 | 1: 0 + 2, 2 + 2, 2 + 5; 2: 4 + 3, 5 + 2, 5 + 6; 3: 6 + 3, 6 + 2, 6 + 6; 4: 6 + 3, 6 + 2, 6 + 6 | 86 | 53 | 8 | 14,968 | 32,648 | 47,616 |
| generic7, pose | 14 + 4 | 18 + 3 | 0 + 3 | 0 + 18 | 1: 0 + 2, 2 + 2, 2 + 5; 2: 4 + 3, 5 + 2, 5 + 6; 3: 6 + 3, 6 + 2, 6 + 6; 4: 6 + 3, 6 + 2, 6 + 6; 5: 6 + 3, 6 + 2, 6 + 6; 6: 6 + 3, 6 + 2, 6 + 6; 7: 6 + 3, 6 + 2, 6 + 6 | 140 | 89 | 14 | 24,208 | 47,768 | 71,976 |
| tiago7, pose | 14 + 4 | 18 + 3 | 0 + 3 | 0 + 15 | 1: 0 + 2, 2 + 2, 2 + 5; 2: 4 + 3, 5 + 2, 5 + 6; 3: 6 + 3, 6 + 2, 6 + 6; 4: 6 + 3, 6 + 2, 6 + 6; 5: 6 + 3, 6 + 2, 6 + 6; 6: 6 + 3, 6 + 2, 6 + 6; 7: 0 + 0, 3 + 2, 3 + 3 | 128 | 80 | 14 | 24,208 | 44,828 | 69,036 |
| tiago7, position | – | – | 0 + 3 | 0 + 15 | 1: 0 + 2, 2 + 2, 2 + 2; 2: 4 + 3, 3 + 2, 3 + 3; 3: 6 + 3, 3 + 2, 3 + 3; 4: 6 + 3, 3 + 2, 3 + 3; 5: 6 + 3, 3 + 2, 3 + 3; 6: 6 + 3, 3 + 2, 3 + 3; 7: 0 + 0, 0 + 2, 0 + 0 | 62 | 52 | 14 | 24,208 | 28,988 | 53,196 |

What every joint's share is divided by (1 + balance² x the mean of |m|² over the joint ranges; balance 3):

- tiago4: 5.54, 5.00, 1.43, 1.64; the largest lever over the reach: 1.00, 0.79, 0.64, 0.27
- generic4: 3.08, 3.66, 1.28, 1.56; the largest lever over the reach: 1.00, 0.75, 0.50, 0.25
- generic7: 2.32, 2.48, 1.78, 1.69, 1.88, 1.09, 1.18; the largest lever over the reach: 1.00, 0.86, 0.71, 0.57, 0.43, 0.29, 0.14
- tiago7: 5.05, 5.03, 1.97, 2.66, 1.01, 1.03, 1.00; the largest lever over the reach: 1.00, 0.84, 0.73, 0.45, 0.24, 0.06, 0.06

The share against the share that removes most (1 + balance² x |m|², the exact value, over the constant it is divided by), over 4000 poses drawn evenly from the joint ranges (seed 20261010, not that of the mean). 1: the best share; above 2: after this joint more is left than before it.

| Arm | Joint | Smallest | Median | Largest | Poses above 2 |
| --- | ---: | ---: | ---: | ---: | ---: |
| tiago4 | 1 | 0.18 | 1.00 | 1.76 | 0.0% |
| tiago4 | 2 | 0.42 | 1.12 | 1.29 | 0.0% |
| tiago4 | 3 | 0.70 | 1.03 | 1.27 | 0.0% |
| tiago4 | 4 | 1.00 | 1.00 | 1.00 | 0.0% |
| generic4 | 1 | 0.32 | 0.72 | 3.24 | 11.2% |
| generic4 | 2 | 0.43 | 0.96 | 1.66 | 0.0% |
| generic4 | 3 | 0.78 | 1.00 | 1.22 | 0.0% |
| generic4 | 4 | 1.00 | 1.00 | 1.00 | 0.0% |
| generic7 | 1 | 0.43 | 0.83 | 3.99 | 5.2% |
| generic7 | 2 | 0.40 | 0.87 | 3.07 | 6.0% |
| generic7 | 3 | 0.56 | 0.91 | 2.18 | 1.1% |
| generic7 | 4 | 0.59 | 0.83 | 2.33 | 5.4% |
| generic7 | 5 | 0.63 | 0.98 | 1.41 | 0.0% |
| generic7 | 6 | 0.92 | 1.00 | 1.08 | 0.0% |
| generic7 | 7 | 1.00 | 1.00 | 1.00 | 0.0% |
| tiago7 | 1 | 0.20 | 0.97 | 1.93 | 0.0% |
| tiago7 | 2 | 0.21 | 1.09 | 1.44 | 0.0% |
| tiago7 | 3 | 0.51 | 1.09 | 1.52 | 0.0% |
| tiago7 | 4 | 0.91 | 1.01 | 1.05 | 0.0% |
| tiago7 | 5 | 0.99 | 1.00 | 1.02 | 0.0% |
| tiago7 | 6 | 1.00 | 1.00 | 1.00 | 0.0% |
| tiago7 | 7 | 1.00 | 1.00 | 1.00 | 0.0% |

## 9. Gazebo as the robot (test 10)

The TIAGo arm in Gazebo with its own trajectory controller and gravity. Physics advances 50 ms for every tick of the network. A joint is told the angle its goal ring holds (ring IK) or the angle the method gives (damped least squares, from the exact joint angles). The pairs are the first test pairs of each arm, the same as on the light robot.

- tiago7 held at 5 poses: the end point from Gazebo's link states (`arm_7_link` in `torso_lift_link`) and the chain's kinematics at the joint angles Gazebo reports differ by 0.22 mm at most, the orientations by 0.145°.
- tiago4 held at 5 poses: the end point from Gazebo's link states (`arm_4_link` in `torso_lift_link`) and the chain's kinematics at the joint angles Gazebo reports differ by 0.14 mm at most, the orientations by 0.018°.

| Arm, task | Robot, method | Trials | Reached | Exact rule reaches | Reached, of those the exact rule reaches | Final position error, median / p90 | In mm, median | Final orientation error, median / p90 | Within the marks after, median | Path / straight, median | Missed (with a joint at a limit) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| tiago7, pose | light robot, ring IK (the same pairs) | 30 | 93% | 97% | 97% | 2.09% / 3.48% | 16.8 | 1.89° / 3.00° | 5.7 s | 1.43 | 2 (0) |
| tiago7, pose | Gazebo, ring IK | 30 | 93% | 97% | 97% | 2.22% / 3.62% | 17.9 | 1.50° / 3.36° | 5.8 s | 1.50 | 2 (0) |
| tiago7, pose | Gazebo, damped least squares | 30 | 100% | 97% | 100% | 0.29% / 0.65% | 2.4 | 0.35° / 0.73° | 4.7 s | 1.01 | 0 (0) |
| tiago4, position | light robot, ring IK (the same pairs) | 20 | 100% | 95% | 100% | 1.62% / 3.15% | 9.9 | – | 4.7 s | 1.40 | 0 (0) |
| tiago4, position | Gazebo, ring IK | 20 | 100% | 95% | 100% | 1.74% / 2.96% | 10.6 | – | 4.9 s | 1.37 | 0 (0) |

| Run | A joint behind its target, median | Trials in which a joint ended more than 3° from its target | The base moved by at most | Reached, of the goals the ring IK reaches on the light robot |
| --- | ---: | ---: | ---: | ---: |
| tiago7, pose: Gazebo, ring IK | 2.30° | 0 of 30 | 12.7 mm | 28 of 28 |
| tiago7, pose: Gazebo, damped least squares | 4.82° | 0 of 30 | 4.5 mm | 28 of 28 |
| tiago4, position: Gazebo, ring IK | 1.10° | 0 of 20 | 7.2 mm | 20 of 20 |
