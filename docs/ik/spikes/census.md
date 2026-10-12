# Spike census of the pose inverse-kinematics network

Written by `scripts/ik_spike_census.py`. tiago4 with a position goal, 35,916 neurons (20,948 of them the inverse-kinematics layer), held at one pose for 2 s of network time; "fire": at least one spike per second; a tick is 50 ms.

## While the hand is asked to move (a goal part of the way to another pose)

| Kind of population | How many | Neurons in each | Fire | Rate of those that fire | Spikes per tick, each | Gaps vary by |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| bias cell of the anchors | 2 | 16 | 16 | 99 Hz | 79 | – |
| cos / sin of a joint | 4 | 120 | 70 | 86 Hz | 301 | 2% |
| forward sheet | 3 | 2304 | 42 | 12 Hz | 25 | – |
| gain pair, one side | 8 | 200 | – | – | 54 | – |
| gate of the forward sheets | 3 | 60 | 30 | 71 Hz | 106 | 3% |
| goal gate | 3 | 60 | 30 | 72 Hz | 107 | 2% |
| goal ring of a joint | 4 | 200 | 22 | 92 Hz | 103 | – |
| goal sheet | 3 | 2304 | 43 | 12 Hz | 27 | – |
| joint ring | 4 | 200 | 27 | 38 Hz | 51 | – |
| length of a pair (cos of pitch) | 1 | 120 | 58 | 87 Hz | 251 | 3% |
| level cells of a sheet | 6 | 60 | 30 | 64 Hz | 95 | 3% |
| output ring | 3 | 100 | 42 | 94 Hz | 197 | – |
| product, comparison and cascade | 38 | 200 | 114 | 71 Hz | 407 | 4% |
| product, orientation of the links | 26 | 200 | 111 | 82 Hz | 454 | 2% |
| ramp readout | 6 | 160 | 89 | 56 Hz | 251 | 7% |
| share half | 8 | 60 | 33 | 70 Hz | 116 | 17% |
| sum: direction c of a joint | 11 | 60 | 30 | 72 Hz | 110 | 3% |
| sum: entry of a link's orientation | 10 | 60 | 31 | 88 Hz | 135 | 4% |
| sum: lever of a joint | 9 | 60 | 31 | 72 Hz | 110 | 3% |
| sum: position cell | 3 | 60 | 30 | 71 Hz | 106 | 2% |
| sum: the offset d | 3 | 60 | 37 | 54 Hz | 99 | 26% |
| sum: what is left of the offset | 11 | 60 | 31 | 60 Hz | 95 | 12% |

All together 39,518 spikes per tick, 790,358 per second: 22.0 per neuron and second. Shares (deg): -4.8, +0.3, -1.5, -0.9. Lead of the goal rings (indices): -6.2, +0.4, -0.8, -1.1. Offset held (% of the reach): +8.1, -5.7, -2.7.

## At the goal (nothing is asked)

| Kind of population | How many | Neurons in each | Fire | Rate of those that fire | Spikes per tick, each | Gaps vary by |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| bias cell of the anchors | 2 | 16 | 16 | 99 Hz | 79 | – |
| cos / sin of a joint | 4 | 120 | 70 | 86 Hz | 301 | 2% |
| forward sheet | 3 | 2304 | 42 | 12 Hz | 25 | – |
| gain pair, one side | 8 | 200 | – | – | 54 | – |
| gate of the forward sheets | 3 | 60 | 30 | 71 Hz | 106 | 2% |
| goal gate | 3 | 60 | 30 | 71 Hz | 106 | 2% |
| goal ring of a joint | 4 | 200 | 23 | 91 Hz | 105 | – |
| goal sheet | 3 | 2304 | 42 | 12 Hz | 25 | – |
| joint ring | 4 | 200 | 28 | 37 Hz | 51 | – |
| length of a pair (cos of pitch) | 1 | 120 | 58 | 86 Hz | 251 | 3% |
| level cells of a sheet | 6 | 60 | 29 | 63 Hz | 93 | 3% |
| output ring | 3 | 100 | 42 | 94 Hz | 196 | – |
| product, comparison and cascade | 38 | 200 | 115 | 71 Hz | 406 | 4% |
| product, orientation of the links | 26 | 200 | 111 | 82 Hz | 454 | 2% |
| ramp readout | 6 | 160 | 90 | 56 Hz | 252 | 8% |
| share half | 8 | 60 | 33 | 64 Hz | 106 | 16% |
| sum: direction c of a joint | 11 | 60 | 30 | 73 Hz | 110 | 3% |
| sum: entry of a link's orientation | 10 | 60 | 31 | 87 Hz | 134 | 4% |
| sum: lever of a joint | 9 | 60 | 30 | 72 Hz | 110 | 2% |
| sum: position cell | 3 | 60 | 30 | 71 Hz | 106 | 2% |
| sum: the offset d | 3 | 60 | 36 | 52 Hz | 94 | 25% |
| sum: what is left of the offset | 11 | 60 | 32 | 58 Hz | 93 | 12% |

All together 39,334 spikes per tick, 786,684 per second: 21.9 per neuron and second. Shares (deg): +0.4, +0.2, -0.1, -0.5. Lead of the goal rings (indices): +0.4, +0.2, -0.1, -0.6. Offset held (% of the reach): -0.4, +0.7, -0.0.
