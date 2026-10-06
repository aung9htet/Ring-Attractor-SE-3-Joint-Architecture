# Technical reference: the pose of the arm from joint rings (SE(3))

The mathematics behind every part that this repo adds, with the derivations. How to run
things is in [`README.md`](README.md); which part is which file is in
[`docs/architecture.md`](docs/architecture.md); what was measured is in
[`docs/results.md`](docs/results.md).

This page builds on two others and does not repeat them:

- the ring attractor, its neuron and its synapses:
  `Ring-Attractor-SO-2-Joint-Architecture/TECHNICAL.md`, sections 2 to 6;
- the orientation of every link, populations that hold a value, readouts, the population
  that multiplies: `Ring-Attractor-SO-3-Joint-Architecture/TECHNICAL.md`, sections 2 to 12.

Two kinds of statement are kept apart: **Built** (what the code does, with its file) and
**Model** (a few equations that explain why it behaves so).

## Contents

1. [The task](#1-the-task)
2. [The position is a weighted sum](#2-the-position-is-a-weighted-sum)
3. [The velocity of the end point](#3-the-velocity-of-the-end-point)
4. [Scaling: what the populations hold](#4-scaling-what-the-populations-hold)
5. [The attractor sheet](#5-the-attractor-sheet)
6. [The anchor](#6-the-anchor)
7. [Switching the anchor](#7-switching-the-anchor)
8. [Path integration](#8-path-integration)
9. [Reading the position](#9-reading-the-position)
10. [A moving arm: why velocity does not remove the delay, and what does](#10-a-moving-arm-why-velocity-does-not-remove-the-delay-and-what-does)
11. [Why path integration is crude here](#11-why-path-integration-is-crude-here)
12. [Measuring](#12-measuring)
13. [Symbols](#13-symbols)

## 1. The task

The pose of the end point is an element of SE(3): a rotation and a translation,

```math
T = \begin{pmatrix} R_{ee} & p \\ 0 & 1 \end{pmatrix}, \qquad R_{ee} \in SO(3),\; p \in \mathbb{R}^3 .
```

The SO(3) network gives $R_k$ for every link $k$ from the joint rings. This repo adds $p$,
and holds it on attractor sheets.

The chain: joint $k$ sits in link $k-1$ at the fixed offset $o_k$ (a vector in that link's
frame) and turns about one axis of its own frame; the end point sits in link $n$ at
$o_{n+1}$. $R_0 = I$.

## 2. The position is a weighted sum

**Derivation.** The origin $c_k$ of joint $k$ is reached from the origin of joint $k-1$ by
going the offset $o_k$, which is given in the frame of link $k-1$, so it has to be turned by
that link's orientation:

```math
c_k = c_{k-1} + R_{k-1}\, o_k, \qquad c_0 = 0 .
```

Unrolling,

```math
c_k = \sum_{m=0}^{k-1} R_m\, o_{m+1}, \qquad
p = c_{n+1} = \sum_{m=0}^{n} R_m\, o_{m+1} .
```

Written by components, $p_i = \sum_m \sum_j (R_m)_{ij}\,(o_{m+1})_j$: every coordinate of the
position is a sum of matrix entries of the links, each multiplied by a **constant** (a link
offset). The SO(3) network already holds every $(R_m)_{ij}$. So the position needs **no
product**: one population per coordinate that sums what is there, with the offsets as
weights (SO-3 reference, section 7).

The first term, $R_0 o_1 = o_1$, is a constant: the origin of the first joint. It is the
centre of the workspace (section 4).

**The lever of joint $k$** is the vector from the joint's origin to the end point:

```math
l_k = p - c_k = \sum_{m=k}^{n} R_m\, o_{m+1} ,
```

also a weighted sum, and $l_1 = p - o_1$.

`ring_se3/kinematics.py` is the exact version, `ring_se3/wiring.py` the list of sums;
`tests/test_kinematics.py` checks them against each other to $10^{-9}$.

## 3. The velocity of the end point

**Derivation.** Hold all joints but joint $k$ still. Everything beyond joint $k$ is then one
rigid body turning about the axis $a_k$ (a unit vector in the base frame) through the point
$c_k$, at the angular speed $\dot q_k$. A point $x$ of a rigid body turning with angular
velocity $\omega$ about a point $c$ moves with $\omega \times (x - c)$. For the end point,

```math
\frac{\partial p}{\partial q_k}\,\dot q_k = \dot q_k\; a_k \times (p - c_k) = \dot q_k\; a_k \times l_k .
```

The position depends on all joints, and the chain rule adds the contributions:

```math
v = \dot p = \sum_{k=1}^{n} \dot q_k\; \big(a_k \times l_k\big) = J(q)\,\dot q, \qquad
J_{\cdot k} = a_k \times l_k .
```

$J$ is the position Jacobian. Its column $k$ says in which direction and how far the end
point moves per radian of joint $k$.

**What it costs.** $a_k$ is a column of $R_k$ (SO-3 reference, section 3) and $l_k$ a sum
(section 2); both exist. New are the products:

```math
w_k = \dot q_k\, a_k \;\;(3\ \text{products}), \qquad
(w_k \times l_k)_i = w_{k,j}\, l_{k,m} - w_{k,m}\, l_{k,j} \;\;(2\ \text{products per component, 6 in all}),
```

nine per joint, with $(i, j, m)$ a cyclic order of $(x, y, z)$. Joint 1 of the chains here
turns about a fixed axis, so its $w_1$ is a constant times $\dot q_1$ and costs nothing: 8,
17, 26, 35, 44, 53 products for 2 to 7 joints. The check against finite differences,
$J_{\cdot k} \approx [p(q + h e_k) - p(q - h e_k)]/2h$, is in the tests.

**The same products serve inverse kinematics.** $J^\top e$ for a position error $e$ is
$(a_k \times l_k)\cdot e$ per joint ([`docs/future-ik.md`](docs/future-ik.md)).

## 4. Scaling: what the populations hold

A population holds numbers in $[-1, 1]$, so every quantity is divided by a bound.

| Quantity | Held as | Bound |
| --- | --- | --- |
| position | $u = (p - o_1)/L$ | reach $L = \sum_{m\ge1} \lVert o_{m+1} \rVert$: the summed link lengths after the first joint |
| lever of joint $k \ge 2$ | $l_k / L_k$ | $L_k = \sum_{m \ge k} \lVert o_{m+1} \rVert$ |
| joint velocity | $\dot q_k / \dot q_{\text{ref}}$ | $\dot q_{\text{ref}} = 1$ rad/s (clipped beyond) |
| end-point velocity | $\hat v = v / (L\,\dot q_{\text{ref}}\,\nu)$ | $\nu = 1.5$ |

*Why the bounds hold.* $\lVert R_m o_{m+1}\rVert = \lVert o_{m+1}\rVert$ because a rotation
preserves length; the triangle inequality gives $\lVert p - o_1\rVert \le L$ and
$\lVert l_k\rVert \le L_k$. Every component of a unit vector $a_k$ is at most 1.

With these, section 3 reads

```math
\hat v = \sum_k \frac{L_k}{L\,\nu}\; \Big(\tfrac{\dot q_k}{\dot q_{\text{ref}}}\, a_k\Big) \times \frac{l_k}{L_k} .
```

For the 7-joint TIAGo arm $L = 0.806$ m, and $\hat v = 1$ is $1.5$ reaches per second.

## 5. The attractor sheet

**Built** (`ring_se3/sheet.py`, `ring_se3/nest/sheet.py`). A sheet is a square of
$n \times n$ neurons ($n = 48$) over one coordinate plane. Neuron $(i, j)$ stands for the
place

```math
c_{ij} = \Big(-E + (i + \tfrac12)\,\delta,\; -E + (j + \tfrac12)\,\delta\Big), \qquad
\delta = \frac{2E}{n} = 0.05, \qquad E = 1.2 ,
```

in scaled coordinates: one *cell* is $\delta = 5\%$ of the reach. $E > 1$ leaves room for a
packet at the edge of the workspace. Three sheets hold the planes $(x, y)$, $(x, z)$,
$(y, z)$.

**Neuron.** The ring neuron of the SO-2 repo, resting exactly at threshold (SO-2 reference,
section 2), as NEST's `iaf_psc_alpha_multisynapse` so that it can have four synapse types:
the ring's excitatory (27 ms) and inhibitory (50 ms) ones, a fast inhibitory one for the
velocity cells (3 ms), and one for the anchor (20 ms).

**Recurrent weights.** The ring's profile $w(d) = 3\,(2e^{-d^2/50} - e^{-d^2/200})$, over the
distance in the plane:

```math
W_{a \to b} = w\big(\lambda\,\lVert c_b - c_a - \kappa\,\delta\, e_a \rVert / \delta\big), \qquad
\lambda = 1.5,\quad \kappa = 3 ,
```

with $\lambda$ the profile's distance units per cell and $e_a$ the unit vector of neuron
$a$'s *label* (below). For $\kappa = 0$ this is the ring's connectivity turned into a disc:
neurons within $6.8/\lambda \approx 4.5$ cells excite each other, a surround inhibits, far
neurons are not connected.

**Model: one packet, anywhere.** As on the ring, neurons inside a group keep each other
firing and silence their surround, and since $W$ depends only on the difference of places
(for $\kappa = 0$), the packet can sit at any place of the plane: a two-dimensional
continuous attractor. Measured: one packet about six cells across, about 25 spikes per
50 ms tick, alive without any input.

**Labels.** Every neuron carries one of four labels, $+a, -a, +b, -b$ for a sheet over the
plane $(a, b)$, in a $2\times2$ tiling. The centre of its *outgoing* weights is moved
$\kappa = 3$ cells along its label ($e_a$ above). Over a packet the four groups are equally
many and their shifts cancel:

```math
\sum_{\text{labels}} e = (1,0) + (-1,0) + (0,1) + (0,-1) = 0 ,
```

so the packet rests. Section 8 breaks that balance on purpose.

## 6. The anchor

The anchor has to pull the packet to a given place $x$ (two scaled coordinates), using only
sums of values that populations hold.

**Built** (`nest/sheet.py: connect_anchor`, `connect_anchor_bias`). The neuron at place $c$
receives the current

```math
I_c(x) = b - \frac{g}{2}\,\lVert c - x \rVert^2 ,
```

a bowl turned upside down with its top $b$ at $x$. Expanding the square,

```math
I_c(x) = \underbrace{g\Big(c \cdot x - \tfrac12 \lVert x \rVert^2\Big)}_{\text{moving part}}
\;+\; \underbrace{\Big(b - \tfrac{g}{2}\lVert c \rVert^2\Big)}_{\text{fixed part}} .
```

- **Moving part.** $c\cdot x - \tfrac12\lVert x\rVert^2 = \sum_{\text{2 coordinates}} \big(c_a x_a - \tfrac12 x_a^2\big)$:
  per coordinate, a weighted sum of $x_a$ and of $x_a^2$. Both are read from one *gate*
  population that holds $x_a$, with two sets of decoders (SO-3 reference, section 6: a
  population's rates can be read as any smooth function of its value, here $x$ and $x^2$).
  The weight from gate neuron $j$ to the sheet neuron at $c$ is
  $g\,(c_a\, d^{(1)}_j - \tfrac12 d^{(2)}_j)/(e\,\tau_{\text{anchor}})$.
- **Fixed part.** The same for every pose: a bias per neuron, delivered by 16 *bias cells*
  that fire at a steady 99 Hz (section 7 says why not as a plain bias current).

No place code is needed on the way in: three numbers go in and a packet comes out. This is
the cosine drive of the SO(3) output rings (SO-3 reference, section 12) done for a plane:
there $x\cos\theta + y\sin\theta$ peaks at the right angle, here $c\cdot x - \tfrac12\lVert c\rVert^2$
peaks at the right place.

**Why the term $-\tfrac12\lVert x\rVert^2$ must be there.** Without it the current would be
$g\,c\cdot x - \tfrac g2\lVert c\rVert^2 + b = b - \tfrac g2\lVert c - x\rVert^2 + \tfrac g2\lVert x\rVert^2$:
still peaked at $x$, but lifted by $\tfrac g2\lVert x\rVert^2$ everywhere. These neurons rest
at threshold, so a positive current makes them fire: a whole disc of radius $\lVert x\rVert$
around $x$ would be driven. With the term the current is at most $b$ and positive only
within $\sqrt{2b/g}$ of $x$.

**Values.** $g = 8$ pA per cell² (3,200 pA per scaled unit²), $b = 32$ pA: the current is
positive within $\sqrt{2\cdot 32/8} \approx 2.8$ cells of $x$, about the size of a packet,
and strongly negative further away. A packet somewhere else is silenced, and a new one
forms at $x$.

**Model: the packet follows like a ball in a bowl.** A packet at $y$ sees a current that
falls off across it with the slope $\nabla I = g\,(x - y)$. The neurons on the side toward
$x$ are driven more, those on the other side less; the packet gains neurons on one side and
loses them on the other, at a speed that grows with the slope:

```math
\dot y = \mu\, g\,(x - y) \quad\Longrightarrow\quad \text{time constant}\;\; \tau_a = \frac{1}{\mu g} .
```

Measured on a sheet alone, with the anchor current set directly and moving at 0.5 units/s:
the packet is 15 ms behind for $g = 8$ and 40 ms for $g = 4$, about inverse to $g$ as the
model says. At rest it sits within 0.14 cells of $x$ in the median.

**An error of the square moves the whole bowl.** If the decoded $x^2$ is off by
$\epsilon$, the current changes by $-\tfrac g2\epsilon$ at *every* neuron: the place of the
peak is untouched, the height is not. With $g = 3{,}200$ pA per unit² and a readout of the
square that is right to 0.015 (root mean square), the height varies by some tens of pA,
comparable to $b$. This is why the position degrades under strong noise
([`docs/results.md`](docs/results.md), section 10) although the packet's place is carried
by the linear part.

## 7. Switching the anchor

Two findings, both by measurement.

**Both parts through one synapse.** The moving part has positive and negative
contributions. Through the ring neuron's own two synapses the negative part would decay
with 50 ms and the positive part with 27 ms: whenever the anchor changes, a net negative
current is left over for some tens of milliseconds and silences the packet. The anchor
therefore has its own synapse type that carries both signs with one time constant (20 ms).

**Both parts through spikes.** The moving part and the fixed part are each thousands of pA
at a neuron far from the centre and cancel to a small sum. If the fixed part were a bias
current switched instantly while the moving part fades with its synapse, the mismatch
during those milliseconds is of that size: the whole sheet fires, then falls silent. With
the fixed part delivered by the bias cells' spikes through the same synapse type, both
parts rise and fall with the same time course and their sum stays within $b$.

Switching the anchor off silences the three gate populations and the bias cells; switching
it on restores their bias currents and sets each of those neurons to a fresh potential
between rest and threshold (a neuron that was held far below rest would otherwise return
late, each at its own time).

## 8. Path integration

With the anchor off, the packet should move with the end point's velocity.

**Built** (`nest/calibrate.py`, `nest/sheet.py: connect_velocity_cells`). For each
coordinate and sign there are 8 *velocity cells*. Each spike of a cell for the direction
$+a$ sends a short, strong inhibition (peak 300 pA, 3 ms) to the sheet neurons labelled
$-a$, in the two sheets that hold coordinate $a$.

**Model: why silencing one group moves the packet.** While the group $-a$ is silent, the
remaining shifts no longer cancel: $\sum e = (+1, 0)$ over the three groups that are left.
The packet's outgoing excitation is centred ahead of it in the direction $+a$, so it moves
that way, at a top speed set by $\kappa$ (measured: about 20 cells per second with the group
silenced all the time). If the group is silenced a fraction $D$ of the time, the mean speed
is about $D$ times that.

**The velocity cell: a rate proportional to its input.** A velocity cell is an
`iaf_psc_alpha` neuron with a very long membrane time constant ($\tau_m = 5$ s,
$C = 250$ pF, $V_{th} = 15$ mV, potential bounded below at rest). From the rate formula
(SO-2 reference, section 2), for $I \gg I_{th} = C V_{th}/\tau_m = 0.75$ pA,

```math
T = \tau_m \ln\frac{I}{I - I_{th}} \approx \tau_m \frac{I_{th}}{I} = \frac{C V_{th}}{I}
\quad\Longrightarrow\quad f \approx \frac{I}{C\,V_{th}} :
```

it integrates its input and fires once per $C V_{th} = 3{,}750$ pA·ms of charge. A current
of 45 pA gives 12 Hz, the *full drive*. An ordinary neuron of the network cannot fire that
slowly in a controlled way; this one turns a value into a pulse rate linearly.

**Push-pull about a resting rate.** Let $S(r)$ be the packet's speed when one group is
pulsed at the rate $r$. Measured, $S$ has a dead zone: below a certain rate the packet does
not move at all (it sticks to the lattice of cells). So all velocity cells get a *resting
drive* $r_0$ (0.19 of the full drive) while the arm moves, and a velocity raises the drive
on one side by $\Delta$ and lowers it on the other:

```math
\text{speed} = S(r_0 + \Delta) - S(r_0 - \Delta) \;\approx\; 2\,S'(r_0)\,\Delta \quad \text{for small } \Delta .
```

Around $r_0$ the response is linear and starts at zero velocity. The price is that the
packet is pulsed from all sides whenever the arm moves, which shakes it. While no joint
moves (all $|\dot q_k| < 0.02$ rad/s) the resting drive is off and the cells are silent.

**Calibration: the speed curve and its inverse.** The curve $\Sigma(\Delta)$, packet speed
in cells per second against the drive $\Delta$, is measured once on a sheet alone (13
drives, four directions, three starts). The packet has to move at

```math
\text{speed wanted} = \frac{\hat v_a\; \dot q_{\text{ref}}\,\nu}{\delta} = 30\,\hat v_a \;\; \text{cells/s}
```

for a velocity value $\hat v_a$ (section 4: $\hat v = 1$ is $\nu = 1.5$ units per second, a
cell is $\delta = 0.05$ units). So the drive must be $\Sigma^{-1}(30\,|\hat v_a|)$ with the
sign of $\hat v_a$. The velocity population's decoders are fitted to this function of its
value instead of to the value itself (SO-3 reference, section 6). The curve ends at 9.47
cells per second: a velocity value above 0.32, that is 0.47 reaches per second, cannot be
followed.

## 9. Reading the position

**Built** (`sheet.SheetMap.centre`, `sheet.combine`). From the spike counts $n_c$ of one
sheet,

```math
\hat x = \frac{\sum_c n_c\, c}{\sum_c n_c} ,
```

the centre of the spikes, over one tick, two ticks (a moving arm) or ten (a held pose). For
a packet of even shape around $x$ the weighted mean of the places is $x$, by symmetry.

Each coordinate is on two sheets. With independent errors of the same variance $s^2$ the
mean of two readings has the variance $s^2/2$; their difference has $2s^2$ and is a check
that costs nothing:

```math
\hat p_x = \tfrac12\big(\hat x^{(xy)} + \hat x^{(xz)}\big), \qquad
\text{disagreement}_x = \hat x^{(xy)} - \hat x^{(xz)} .
```

Measured: a single sheet is off by 0.71% to 1.22% of the reach in its plane, the two copies
of a coordinate differ by only 0.30% to 0.34%. The errors are therefore *not* independent:
most of the error is common to the sheets, because it is the error of the position cells
that anchor them all. Averaging over sheets cannot remove it.

Position in metres: $\hat p = o_1 + L\,\hat u$.

## 10. A moving arm: why velocity does not remove the delay, and what does

**Model.** Let the packet obey, along one coordinate,

```math
\dot y(t) = \hat v(t) + \frac{1}{\tau_a}\big(p(t - \tau) - y(t)\big) :
```

pushed by the velocity and pulled toward the anchor, which is the true position of a time
$\tau$ earlier (the delay of the SO(3) network, about 0.2 s).

*Steady motion.* For $p(t) = v t$ and a perfect $\hat v = v$, try $y = v t - \epsilon$:

```math
v = v + \frac{1}{\tau_a}\big(v(t - \tau) - vt + \epsilon\big) \;\Longrightarrow\; \epsilon = v\,\tau .
```

The packet is behind by $v\tau$ **whatever $\tau_a$ is and even with a perfect velocity
input**. Path integration moves the packet at the right speed, but the anchor keeps pulling
it back to where the arm was. Velocity added as a push cannot remove a delay of the anchor.
Measured: with and without path integration the delay is the same (0.23 s and 0.24 s for
the 7-joint arm).

*Anchoring ahead.* Give the anchor the position where the end point will be:

```math
x_{\text{anchor}}(t) = p(t - \tau) + \tau\,\hat v(t - \tau) \approx p(t) ,
```

a first-order prediction over the delay. This is one more weighted sum into the gate
populations (position plus $\tau\,\dot q_{\text{ref}}\,\nu$ times the velocity value), option
`anchor_lead_s`. Measured with $\tau = 0.2$ s: delay 0.03 s, error while moving 2.28% of the
reach instead of 5.26% (7 joints). The same idea at the input of the SO(3) network (writing
the joint angles ahead) removes the delay for the orientation too.

## 11. Why path integration is crude here

**Measured.** A free packet driven by a known velocity moves at 0.66 to 1.71 times the
speed asked for (10th to 90th percentile over 79 runs), and with the anchor off the packets
of a moving arm fall behind or wander ([`docs/results.md`](docs/results.md), sections 2
and 8).

**Model: counting spikes.** A packet has about 30 neurons that fire about 15 times a second
each. One group is a quarter of them. A velocity-cell spike silences that group for a few
milliseconds; whether this removes a spike depends on whether one of those seven or eight
neurons was about to fire. The push per pulse is therefore not a fixed step but zero or one
missing spike, by chance, and the packet's motion is a sum of such events. The speed is
right on average and varies by tens of percent between runs.

Two more effects follow from the layout:

- **Two coordinates share a sheet.** Silencing $-a$ and $-b$ at once leaves two groups
  instead of three; the packet is then smaller and moves differently from what two separate
  calibrations predict (measured during development on a sheet alone: diagonal motion off
  by about 45% of what the one-coordinate curve predicts, in the median).
- **The lattice.** The packet prefers to sit at certain places relative to the cells. A
  weak push does not move it (the dead zone), and when released from the anchor it settles
  to the next such place: about one cell, 5% of the reach.

**What would change it.** A packet with many more neurons, or neurons that fire faster, so
that a push removes a predictable number of spikes; or separate layers per direction in
place of labels inside one sheet. Both are redesigns of the sheet.

## 12. Measuring

- **Position error** $\lVert \hat p - p \rVert$, given as a share of the reach $L$ and in mm.
- **"Similar"**, fixed before the runs: median at most 2% of $L$, 90th percentile at most 4%.
- **Delay**: the shift $\tau$ that minimises the median of
  $\lVert \hat p(t) - p(t - \tau)\rVert$ over the ticks in which the arm moves, refined
  between ticks by a parabola through the three lowest values.
- **Error over distance** (path integration alone): $\lVert \hat p - p \rVert$ at the end of a
  gap divided by how far the end point moved during it; 1 for a packet that stayed where it
  was, 0 for a perfect one.
- **Back after a gap**: the first tick from which the single-tick error is again at the
  level it has with the anchor on in the same run (its 90th percentile) for five ticks.

## 13. Symbols

| Symbol | Meaning | Value |
| --- | --- | --- |
| $o_k$ | origin of joint $k$ in link $k-1$; $o_{n+1}$: the end point in link $n$ | from the chain |
| $c_k$, $p$ | origin of joint $k$, end point, base frame | |
| $l_k$, $a_k$ | lever and axis of joint $k$ | |
| $J$ | position Jacobian, columns $a_k \times l_k$ | |
| $L$, $L_k$ | reach; largest length of lever $k$ | 0.806 m for `tiago7` |
| $u$, $\hat v$ | scaled position, scaled velocity | in $[-1, 1]$ |
| $\dot q_{\text{ref}}$, $\nu$ | joint velocity and velocity scale that are held as 1 | 1 rad/s, 1.5 |
| $n$, $E$, $\delta$ | neurons per side of a sheet, its half-width, one cell | 48, 1.2, 0.05 |
| $\lambda$, $\kappa$ | profile units per cell, shift of the outgoing weights in cells | 1.5, 3 |
| $g$, $b$ | gain and top of the anchor | 8 pA/cell², 32 pA |
| $d^{(1)}$, $d^{(2)}$ | decoders of a gate population for its value and its square | fitted |
| $\tau_a$ | time constant with which a packet follows its anchor | about 15 ms |
| $r_0$, $\Delta$ | resting drive and added drive of the velocity cells | 0.19, up to 0.3 of the full drive |
| $\Sigma(\Delta)$ | speed curve of a packet | measured, up to 9.47 cells/s |
| $\tau$ | delay of the readout behind a moving arm | 0.21 to 0.25 s |
