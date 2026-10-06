# How the network gives the pose

What the network is made of and why. How to run it: the top-level
[`README.md`](../README.md). What it achieves: [`results.md`](results.md).

![the pose network of the 7-joint arm](figures/architecture.png)

Drawn by `scripts/draw_architecture.py`; the counts come from the wiring the network is
built from.

## 1. The task

A chain of revolute joints, each joint angle held as a bump on its own ring attractor. The
SO-3 repo reads the orientation of every link from those rings: a rotation, an element of
SO(3). This repo adds the position of the end point, so that orientation and position
together are the pose: an element of SE(3), the forward kinematics of the arm.

The position is held on three attractor sheets, in the manner of the pose cells of RatSLAM:

| RatSLAM | Here |
| --- | --- |
| pose cells: a packet of activity that holds itself | three sheets of spiking neurons, one packet each, for the planes xy, xz, yz |
| local view cells pull the packet to a known place | the position computed from the rings pulls the packet (the *anchor*) |
| odometry moves the packet | the end point's velocity, computed from the joint velocities, moves the packet (*path integration*) |
| heading | the orientation of the SO(3) network |

The chains are those of the SO-3 repo ([`ring_se3/chains.py`](../ring_se3/chains.py)):
`generic2` .. `generic7` (joints about x, y, z, x, y, z, x, links of 0.2 m) and `tiago2` ..
`tiago7` (the first joints of the TIAGo arm with the offsets of its URDF). A shortened TIAGo
arm ends at the origin of the next joint of the real arm, so that each of its joints moves
the end point; `tiago7` ends at `arm_tool_link`.

## 2. The maths

`R_k` is the orientation of link k (`R_0` the identity), `o_k` the fixed origin of joint k in
link k-1, and `o_(n+1)` the end point in link n.

```
origin of joint k:   c_k = Σ_(m<k)  R_m · o_(m+1)
end point:           p   = Σ_(m<=n) R_m · o_(m+1)
lever of joint k:    l_k = p - c_k = Σ_(m>=k) R_m · o_(m+1)
velocity:            v   = Σ_k  q̇_k · (a_k × l_k)          a_k: the joint's axis, a column of R_k
```

- **The position needs no product.** The SO(3) network already holds every entry of every
  `R_m`; the position is those entries weighted with the link offsets and summed.
- **The velocity needs nine products per joint**: three for `q̇_k · a_k`, six for the cross
  product with the lever. Joint 1 of every chain here turns about a fixed axis, which saves
  its first three.
- **Scaling.** The network holds the position as `(p - centre) / radius`, three numbers in
  [-1, 1]: the centre is the origin of the first joint and the radius the summed length of
  the links after it (the *reach*; 0.806 m for `tiago7`). Every lever is divided by its
  largest possible length, every joint velocity by 1 rad/s, and the velocity of the end point
  by 1.5 reaches per second.

`ring_se3/kinematics.py` is the exact version; `ring_se3/wiring.py` writes the same sums and
products as a list of small steps on top of the SO(3) wiring, which the network and its copy
without spikes both follow. `tests/test_kinematics.py` checks both against each other and
against finite differences.

## 3. The parts

| Part | What it holds | Neurons | Kind |
| --- | --- | ---: | --- |
| SO(3) network | orientation of every link; lift, pitch, yaw rings | 16,740 for 7 joints | unchanged from the SO-3 repo |
| Position cells | x, y, z of the end point | 3 x 60 | a link-type population: a weighted sum of earlier values |
| Lever populations | the lever of each joint from the second on | 3 x 60 per joint | the same |
| Joint-velocity inputs | `q̇_k`, set from outside every tick | 60 per joint | the same, driven by a current |
| Velocity products | `q̇ · axis`, then the cross product | 200 each; 53 for 7 joints | the multiplying population of the SO(3) network |
| Velocity populations | vx, vy, vz of the end point | 3 x 60 | link-type |
| Gates | the position as handed to the sheets | 3 x 60 | link-type; silenced to switch the anchor off |
| Bias cells | the fixed part of the anchor | 16 | ordinary neurons firing at a steady 99 Hz |
| Velocity cells | pulses that move the packets | 6 groups of 8 | slow integrate-and-fire neurons |
| Attractor sheets | one packet each | 3 x 48 x 48 | the SO-2 ring neuron |

For the 7-joint arm: 36,356 neurons, of which 16,740 are the SO(3) network, 12,704 the
position and velocity populations and 6,912 the sheets.

All populations between the rings and the sheets use the readouts of the SO(3) calibration
(a sum, a product). Nothing was fitted to a pose of any chain.

## 4. The sheets

[`ring_se3/sheet.py`](../ring_se3/sheet.py) (geometry, weights, readout),
[`ring_se3/nest/sheet.py`](../ring_se3/nest/sheet.py) (NEST).

**Layout.** A sheet is a square of 48 x 48 neurons over one coordinate plane. It covers
[-1.2, 1.2] of the scaled position in both directions, so that a packet at the edge of the
workspace still fits; one cell is 0.05 of the reach (4 cm for the 7-joint arm). It does not
wrap around: the workspace has an edge.

**Neuron and weights.** The neuron is the ring neuron of the SO-2 repo with the same
parameters, and the recurrent weights are the ring's own profile (near neighbours excite, a
wider surround inhibits), taken over the distance in the plane instead of along the ring.
With these a single packet about six cells across forms and keeps itself alive without
input. That was the first thing tried, and it worked.

**Reading a position.** The centre of a sheet's spikes: of one 50 ms tick, of the last two
ticks (a moving arm), or of the last 0.5 s (a held pose). Each coordinate is on two sheets;
the estimate is the mean of the two and their difference is kept as a check.

**Anchor.** The neuron at place `c` gets the current

```
offset - gain/2 · |c - x|²   =   gain · (c·x - |x|²/2)   +   (offset - gain/2 · |c|²)
```

with `x` the anchored place. It is largest at `x` and falls off in all directions, so the
packet is pulled to `x` like a ball in a bowl. The right-hand side is how it is built: the
first part is a weighted sum of `x` and `x²` per coordinate, both read from a *gate*
population that holds the coordinate (one readout for the value, one for its square); the
second part is the same for every pose and comes from the *bias cells*. No place code is
needed on the way in: three numbers go in, a packet comes out. This is the cosine drive of
the SO(3) output rings, done for a plane.

Both parts reach the sheet through one synapse type with one time constant (20 ms), and the
gates and the bias cells are silenced together to switch the anchor off. Measured on the way
(section 8): with the two parts on different time constants, or switched at different
moments, the packet is destroyed by the switch.

**Path integration.** Every sheet neuron carries one of four labels (+a, -a, +b, -b for a
sheet over the plane ab, in a 2 x 2 tiling), and the centre of its outgoing weights is moved
three cells that way. With all four groups active the shifts cancel and the packet rests.
*Velocity cells* act against this balance: each spike of a cell for the direction +a
silences, for a few milliseconds, the sheet neurons labelled -a; the packet then moves
toward +a. This is the arrangement of grid-cell models with direction-labelled neurons
(Burak and Fiete 2009), with the velocity acting through inhibition.

Three points decide how well this works, all found by measurement (section 8):

- A velocity cell is a slow integrate-and-fire neuron, so its rate is proportional to its
  input, and its synapse on the sheet is fast (3 ms) and strong, so one spike is one short
  push.
- While any joint moves, all six groups of velocity cells get a resting drive, and a
  velocity raises the drive on one side and lowers it on the other. Without the resting
  drive a slow motion does not move the packet at all: it sticks to the cells.
- How fast the packet moves for a given drive is measured once on a sheet alone (the *speed
  curve*), and the readout of the velocity populations is fitted to give, for each
  velocity, the drive that the curve says is needed.

**Persistent state.** A packet stays alive without input: that is the attractor. How well it
keeps its place, and how well path integration carries it, is measured in tests 3 and 8.

## 5. What is taken from where

- **RatSLAM** (Milford, Wyeth and Prasser 2004): the three roles in the table of section 1.
  Different here: RatSLAM's pose cells are one three-dimensional block (x, y, heading) that
  wraps around, its packet is moved by copying activity, and its local view is learned from
  camera images. Here the sheets are two-dimensional and bounded, the packet is moved by
  spikes, and the anchor is computed, not learned. There is no map and no loop closure:
  the arm's workspace is known.
- **NeuroSLAM** (Yu et al. 2019): position in three dimensions. It uses a three-dimensional
  grid of cells; here three two-dimensional sheets hold the three coordinates twice each.
- **Lin, Yiu and Leibold (ICLR 2026)**, "Emergence of Spatial Representation in an
  Actor-Critic Agent with Hippocampus-Inspired Sequence Generator" (the code is named
  `SF_hipposlam`): a shift register that keeps a brief input readable for a number of
  steps. That paper has no attractor network; the persistent packet here is the attractor's
  own. What is taken from it is a small experiment: a shift register keeps where the
  packets were in the last 0.5 s, and the velocity those places show is handed to the
  velocity cells when both the anchor and the joint velocities are taken away
  ([`results.md`](results.md), section 9). The register is a plain list outside NEST, as
  the paper's is a linear register.

## 6. Calibration

Two files in `artifacts/`:

- `calibration_*.npz`: the SO(3) calibration (readouts of a sum, a product, the cos / sin
  population), copied from the SO-3 repo; `scripts/calibrate.py` writes it again.
- `pose_*.npz`, written by `scripts/calibrate_pose.py`
  ([`ring_se3/nest/calibrate.py`](../ring_se3/nest/calibrate.py)): the gate readout (value
  and square), the rate of a bias cell as NEST runs it, the speed curve of a packet (13
  drives, four directions, three starts each), and the velocity readout fitted to the
  inverse of that curve.

The few numbers of the sheet itself (`SheetDesign` in `ring_se3/sheet.py`: size, packet
width, shift, anchor gain, the velocity cells) were set on sheets alone and on a 3-joint
chain with poses that are not test poses, and frozen before the test poses were run.

## 7. Running

`PoseNetwork.step(q, qdot)` ([`ring_se3/nest/network.py`](../ring_se3/nest/network.py))
writes joint angles and joint velocities, runs one 50 ms tick and returns the orientation
(as the SO(3) network does), the position from the sheets, the position the position cells
hold, the velocity the velocity populations hold, and the sheets' spikes.
`set_anchor(on)` and `set_path_integration(on)` are the two switches. Option
`anchor_lead_s` hands the gates the position plus velocity times that many seconds.

## 8. What changed from the plan, and why

Decided by measurement during the work; each is covered in [`results.md`](results.md).

- **Sheet size by measurement, not by a copy without spikes.** What limits a sheet (the
  packet settling on its cells, drifting, being pushed by single spikes) is not in a rate
  picture. The copy without spikes covers the position and velocity populations only.
  Sizes tried on a sheet alone: 32 x 32 with a packet three cells across, and 48 x 48 with
  a packet six cells across. The larger packet moves more evenly.
- **Path integration: four attempts.** (1) Velocity as an added current on the labelled
  groups: the whole sheet fires, because these neurons rest exactly at threshold.
  (2) Velocity as a steady inhibition of the opposing group: nothing moves below a
  threshold and everything at full speed above it. (3) Inhibition in pulses from cells
  firing in proportion to the speed: a graded speed, but nothing below a certain rate.
  (4) The same in push-pull about a resting rate: a graded speed from zero. This is what
  is built. It is still crude (section 0 of the tables).
- **A third and a fourth synapse on the sheet neuron**: a fast one for the velocity cells'
  pulses, and one for the anchor that carries both signs with one time constant. The neuron
  is otherwise the SO-2 ring neuron.
- **Bias cells.** The fixed part of the anchor was first a current set in the sheet
  neurons. Switching the anchor then left one part behind for a few milliseconds, which
  made the whole sheet fire and then fall silent. With the fixed part delivered by spikes
  through the same synapse both parts fade together.
- **The resting drive of the velocity cells is given from outside**: it is on while any
  joint moves faster than 0.02 rad/s. In the animal this would be a speed signal.
- **Shift-register experiment**: the register is outside NEST (section 5).

## 9. Code taken from the SO-3 repo

`ring_so3/` is a copy of that package from branch `fk-chain` of
`Ring-Attractor-SO-3-Joint-Architecture` as it stood on 2026-10-05 (uncommitted there),
unchanged, so that this repo runs on its own. `scripts/calibrate.py`, `docker/` and the
SO(3) calibration files are copied too. `ring_se3/` and everything else are new;
`ring_se3/sim/page.py` is the SO(3) page with the position added, and
`ring_se3/sim/server.py` extends the SO(3) server.
