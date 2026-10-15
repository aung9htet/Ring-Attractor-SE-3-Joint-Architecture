# Ring attractor kinematics: the models of the three repos in one account

One document for the three repos:

| Stage | Repo | What it builds |
| --- | --- | --- |
| SO(2) | `Ring-Attractor-SO-2-Joint-Architecture` | one ring attractor per joint, and a spiking controller that moves the joint to a goal |
| SO(3) | `Ring-Attractor-SO-3-Joint-Architecture` | the orientation of every link of an arm from the joint rings, and an inverse kinematics for the orientation of the hand |
| SE(3) | `Ring-Attractor-SE-3-Joint-Architecture` | the position of the end point on attractor sheets, and an inverse kinematics for the full pose |

It is written so that each part can be lifted and derived on its own. Every part has the
same six items:

1. **What it is for.**
2. **Mathematics**: definitions and the derivation, step by step, with the assumptions named.
3. **In spikes**: which neurons carry each quantity, what makes them fire, how the equation
   becomes weights, currents and rates, and where the spiking version departs from it.
4. **Built**: the values used and the file that implements it.
5. **Measured**: what the model predicts beside what was measured, with the source.
6. **To derive further**: what is a model and not a proof, or is known only by measurement.

Two kinds of statement are kept apart throughout. **Built** is what the code does.
**Model** is a few equations that explain why the built thing behaves as it does; the
network that runs is always the spiking one.

Each repo has its own technical reference with its own section numbers and symbols. This
document uses one notation; the last section maps it to each repo's. Numbers are copied from
the repos' generated tables and data files; the last section says from which.

## Contents

- **Part 0. The idea**: [0.1](#01-the-claim) the claim · [0.2](#02-the-three-stages) the three stages · [0.3](#03-notation) notation
- **Part A. Common ground**: [A.1](#a1-the-neuron) the neuron · [A.2](#a2-the-synapse-what-a-weight-is) the synapse · [A.3](#a3-three-kinds-of-signal) three kinds of signal · [A.4](#a4-a-population-that-holds-a-value) a population that holds a value · [A.5](#a5-reading-a-value-and-passing-it-on) reading and passing on · [A.6](#a6-a-population-that-multiplies) multiplying · [A.7](#a7-what-a-derivation-becomes-in-spikes) from a derivation to spikes
- **Part B. SO(2)**: [B.1](#b1-the-task-joint-angle-and-ring-index) task and mapping · [B.2](#b2-the-ring-attractor) the ring attractor · [B.3](#b3-writing-an-angle) writing · [B.4](#b4-reading-an-angle-the-population-vector) reading · [B.5](#b5-the-fourier-readout) Fourier readout · [B.6](#b6-the-ramp-comparator) ramp comparator · [B.7](#b7-the-gain-pair-moving-a-bump) gain pair · [B.8](#b8-the-decoder-spikes-to-joint-velocity) decoder · [B.9](#b9-two-rings-open-loop-v3) v3 · [B.10](#b10-three-rings-closed-loop-v4) v4 · [B.11](#b11-time-the-baseline-and-scoring) time, baseline, scoring
- **Part C. SO(3)**: [C.1](#c1-the-orientation-is-a-sum-of-products) sum of products · [C.2](#c2-joint-by-joint) joint by joint · [C.3](#c3-from-a-bump-to-cos-and-sin) bump to cos and sin · [C.4](#c4-lift-pitch-yaw-and-the-output-rings) output rings · [C.5](#c5-the-error-and-how-it-grows-along-the-arm) error · [C.6](#c6-a-moving-arm-delay-and-writing-ahead) delay · [C.7](#c7-comparing-two-orientations) comparison · [C.8](#c8-the-cascade) cascade · [C.9](#c9-the-cascade-as-populations) as populations · [C.10](#c10-a-joints-goal-ring) goal ring · [C.11](#c11-closing-the-loop) the loop · [C.12](#c12-the-robots) robots · [C.13](#c13-the-standard-methods-and-where-the-cascade-sits) standard methods
- **Part D. SE(3)**: [D.1](#d1-the-position-is-a-weighted-sum) position · [D.2](#d2-the-velocity-of-the-end-point) velocity · [D.3](#d3-the-attractor-sheet) sheet · [D.4](#d4-the-anchor) anchor · [D.5](#d5-path-integration) path integration · [D.6](#d6-reading-the-position-delay) reading · [D.7](#d7-the-pose-error-and-what-a-joint-does-to-it) pose error · [D.8](#d8-the-cascade-over-turn-and-offset) pose cascade · [D.9](#d9-dividing-by-a-constant) constant divisor · [D.10](#d10-the-balance) balance · [D.11](#d11-from-a-packet-to-a-number-the-ramp-readout) ramp readout · [D.12](#d12-level-cells) level cells · [D.13](#d13-the-layer-as-populations) as populations · [D.14](#d14-closing-the-loop-on-a-pose) the loop · [D.15](#d15-gazebo-for-the-pose) Gazebo
- **Part E. Together**: [E.1](#e1-from-the-mathematics-to-the-spikes-one-table) one table · [E.2](#e2-what-is-reused-and-what-each-stage-adds) reuse · [E.3](#e3-sizes-and-speed) sizes · [E.4](#e4-what-is-shown-and-what-is-not) claims
- **Back matter**: [F.1](#f1-notation-of-this-document-against-the-three-references) notation map · [F.2](#f2-where-the-numbers-come-from) sources

---

# Part 0. The idea

## 0.1 The claim

A robot arm's state and its goal can be held, compared and turned into joint motion by one
kind of machinery, attractor networks of spiking neurons, without a solver.

- **State is a place code on an attractor.** A joint angle is *where* a bump of activity
  sits on a ring of neurons. A position in space is *where* a packet sits on a sheet of
  neurons. The network is the same after a shift along the ring or sheet, so the bump can
  rest anywhere: a continuous attractor. It stays without input, which makes it a memory.
- **Between attractors, a value is a weighted sum of firing rates.** Cosines, matrix
  entries, products and errors are held by populations in which about half of the neurons
  fire whatever the value is; the value is in a weighted sum of their rates. Two operations
  are available: adding (currents add in a membrane) and multiplying two values (one
  population with a suitable readout). There is no division and no matrix inverse.
- **Kinematics is sums and products.** The orientation of a link is a sum of products of
  the cosines and sines of the joint angles (C.1). The position of the end point is a
  weighted sum of link orientations with the link offsets as weights (D.1). What a joint
  does to the hand is its axis and its axis crossed with its lever (D.7). All of it fits
  the two operations.
- **Motion comes from a comparison.** The goal is held the same way as the state. Comparing
  them gives an error; the error is handed to the joints as *shares*; a share moves a bump;
  the joint follows the bump. The loop closes through the robot and the state readout.
- **Nothing is inverted.** Each joint takes the part of the error it can remove and passes
  on the rest (C.8, D.8). The sequence of small steps is the trajectory.

## 0.2 The three stages

| | SO(2) | SO(3) | SE(3) |
| --- | --- | --- | --- |
| State held | one joint angle | orientation of every link | position of the end point as well |
| Held on | a ring per joint | the rings, read into populations; three output rings | three sheets (planes xy, xz, yz) |
| Goal | a goal ring per joint | three goal rings (lift, pitch, yaw) | three goal sheets as well |
| Comparison | ramp comparator, far; projection, near | $e = \tfrac12\sum_i r_i \times g_i$, products | goal ramps minus hand ramps, on the sheets |
| From error to motion | a gain pair moves the bump; the same spikes move the joint | cascade of shares down the arm; a gain pair writes each joint's goal ring | the same cascade over turn and offset |
| Loop closed by | state ring fed by the joint (v4) | forward network fed by the robot | forward network and forward sheets |
| New mechanisms | attractor ring, comparator, gain pair, predictive ring | value populations, product population, share halves, written goal ring | attractor sheet, anchor, velocity cells, level cells, ramp readout |

What each stage keeps from the one before: SO(3) keeps the ring, its neuron, its encoder and
the gain pair, and gives up the comparator in favour of a graded error. SE(3) keeps all of
SO(3) and adds a second attractor, the sheet, with what is needed to write a number onto it
(the anchor) and to read a number back from it (the ramp readout with level cells). The ramp
of the SO(2) comparator returns there, in a plane.

## 0.3 Notation

Vectors are in the base frame unless said otherwise. A tick is 50 ms.

| Symbol | Meaning | Unit or value |
| --- | --- | --- |
| $n$ | number of joints | |
| $q_k$, $\dot q_k$, $q^\ast$ | angle and velocity of joint $k$; a goal angle | rad, rad/s |
| $N$ | neurons on a ring | 200 |
| $\theta_i = 2\pi i/N$ | ring angle of neuron $i$ | |
| $i(q)$, $\rho_q$ | ring index of a joint angle; radians per ring index | |
| $\varphi$ | position of a bump on a ring | |
| $w(d)$ | recurrent weight at distance $d$ along a ring or sheet | |
| $C_m, \tau_m, V_{th}, t_{\text{ref}}$ | membrane capacitance, time constant, threshold, refractory time | |
| $I_{th} = C_m V_{th}/\tau_m$ | rheobase: the smallest current that makes a neuron fire | |
| $J$, $f(J)$ | input current of a neuron, its rate for a constant current | pA, Hz |
| $\tau_{\text{syn}}$ | time constant of a synapse | ms |
| $r_j$ | firing rate of neuron $j$ | Hz |
| $\epsilon_i$, $\alpha_i$, $\beta_i$ | encoder, gain and bias of population neuron $i$ | |
| $\varrho$ | radius of the range a population covers | 1 unless said |
| $\delta_j$ | decoder (readout weight) of neuron $j$ | |
| $R_k$, $R$, $r_i$ | orientation of link $k$; of the hand; column $i$ of $R$ | |
| $A_k$ | fixed rotation of joint $k$'s frame in link $k-1$ | |
| $a_k$ | turning axis of joint $k$ | unit vector |
| $\phi, \vartheta, \psi$ | lift, pitch, yaw | rad |
| $G$, $g_i$ | goal orientation; its column $i$ | |
| $\Delta$, $\hat n$ | angle and axis of the rotation from $R$ to $G$ | |
| $e$ | the turn still to make, $\sin\Delta\,\hat n$ | rad |
| $o_k$ | origin of joint $k$ in link $k-1$; $o_{n+1}$: the end point in link $n$ | m |
| $p$, $p^\ast$ | end point; goal position | m |
| $l_k$ | lever of joint $k$: from its origin to the end point | m |
| $L$, $L_k$ | reach; the largest length of lever $k$ | m |
| $u = (p - o_1)/L$ | scaled position | in $[-1,1]^3$ |
| $d = (p^\ast - p)/L$ | the offset still to make | reaches |
| $m_k = a_k \times l_k / L$ | how far joint $k$ moves the end point per radian | reaches/rad |
| $c_k = a_k \times l_k / L_k$, $\rho_k = L_k/L$ | direction of that move, and its largest size | |
| $w$ | balance: radians of turn that one reach of offset counts as | 3 |
| $E = (e;\, w\,d)$, $J_k = (a_k;\, w\,m_k)$ | stacked error; what joint $k$ does to it per radian | |
| $t_k$, $\bar t_k$ | share of joint $k$; the largest share it takes | rad |
| $N_k$, $\omega_k$ | what a share is divided by; relaxation factor $\lVert J_k\rVert^2/N_k$ | |
| $h_k^+$, $h_k^-$ | the two halves of a share | in $[0,1]$ |
| $r_R$, $r_L$ | rates of the right and the left side of a gain pair | Hz |
| $\ell$, $m$ | lead of a goal ring over its state ring; shift of the gain pair's writing | ring indices; 8 |
| $\kappa$ | loop gain | 1/s |
| $\tau$ | delay of a loop | s |
| $\mathcal R_a$, $\mathcal S$ | ramp and strength of a sheet's spikes | |
| $K$ | gain of the level-cell loop | |

---

# Part A. Common ground

## A.1 The neuron

**What it is for.** Every neuron in the three repos is NEST's `iaf_psc_alpha`: a leaky
integrator with a threshold. Everything else follows from two of its properties: the
smallest current that makes it fire, and how its rate grows above that.

**Mathematics.** With the resting potential and the reset at zero,

```math
C_m \frac{dV}{dt} = -\frac{C_m}{\tau_m}\,V + J(t), \qquad V \ge V_{th} \;\Rightarrow\; \text{spike},\; V \leftarrow 0,\; \text{silent for } t_{\text{ref}} .
```

*Rate for a constant current.* After a spike (and the refractory time) the potential starts
from zero. The solution of the linear equation for constant $J$ is

```math
V(t) = \frac{J\,\tau_m}{C_m}\left(1 - e^{-t/\tau_m}\right) .
```

It approaches $J\tau_m/C_m$ from below. It reaches $V_{th}$ only if $J\tau_m/C_m > V_{th}$,
that is if

```math
J > I_{th} = \frac{C_m V_{th}}{\tau_m} \qquad \text{(the rheobase)} .
```

Setting $V(T) = V_{th}$ and solving for $T$:

```math
1 - e^{-T/\tau_m} = \frac{I_{th}}{J}
\;\;\Longrightarrow\;\;
T = \tau_m \ln\frac{J}{J - I_{th}}, \qquad
f(J) = \frac{1}{t_{\text{ref}} + \tau_m \ln\dfrac{J}{J - I_{th}}} \quad (J > I_{th}),
```

and $f = 0$ otherwise. The inverse is $J(f) = I_{th}\big/\big(1 - e^{-(1/f - t_{\text{ref}})/\tau_m}\big)$.

*A neuron resting exactly at threshold.* Give the neuron a constant bias equal to its
rheobase and write any further input as a fraction of it, $J = I_{th}(1 + \varepsilon)$.
With $t_{\text{ref}} = 0$,

```math
f = \frac{1}{\tau_m \ln\!\big((1+\varepsilon)/\varepsilon\big)} \;\approx\; \frac{1}{\tau_m \ln(1/\varepsilon)} \quad (\varepsilon \ll 1) .
```

Three consequences:

- any net positive input makes the neuron fire, any net negative input silences it: the
  sign of the input decides;
- the rate grows only with the logarithm of the input, so the rate of a neuron in a bump is
  nearly independent of how strong its input is;
- a neuron whose bias is a little below the rheobase can never fire on its own.

**In spikes.** The four kinds of cell, all `iaf_psc_alpha` with different numbers:

| Cell | Where | $C_m$ | $\tau_m$ | $V_{th}$ | Bias | Rheobase | Synapses | $t_{\text{ref}}$ |
| --- | --- | ---: | ---: | ---: | --- | ---: | --- | ---: |
| ring neuron | joint rings, goal rings, sheets | $3.96\cdot10^6$ pF | 7.04 ms | 0.8 mV | 450,000 pA, exactly its rheobase | 450,000 pA | 27 ms (+), 50 ms (−) | 0 |
| gain neuron | the two sides of a gain pair | $3.96\cdot10^6$ pF | 7.04 ms | 1.35 mV | 0 (SO-2); 304,000 pA (SO-3, SE-3) | 759,375 pA | 27 ms, 50 ms | 0 |
| population neuron | everything between rings | 250 pF | 20 ms | 15 mV | its own $\beta_i$ | 187.5 pA | 5 ms | 2 ms |
| readout cell | numbers for tests and pages | 250 pF | 50 ms | out of reach | – | – | 5 ms | – |

Check of the ring neuron: $I_{th} = 3.96\cdot10^6 \cdot 0.8 / 7.04 = 450{,}000$ pA, equal to
its bias.

**Built.** SO-2 `src/config/model_params/neuron_params.json`; SO-3 `ring_so3/neurons.py`,
`ring_so3/nest/ring.py`.

**Measured.** A ring neuron whose spike generator adds 0.147 of the rheobase should fire at
$1/(7.04\ \text{ms}\cdot\ln(1.147/0.147)) = 69$ Hz; measured up to 72 Hz. A population
neuron driven with 397 pA should fire at 68 Hz; measured: one spike every 14.7 ms, 68 Hz.

**To derive further.** $f(J)$ is for a constant current. The real input is a sum of pulses
(A.2); the rate for a fluctuating input is not derived, only measured (which is why every
readout is fitted to measured rates, A.5).

## A.2 The synapse: what a weight is

**What it is for.** To turn "a spike arrives" into "a current flows", and to say what the
number called a weight means.

**Mathematics.** A spike at time $t_s$ through a synapse of weight $\mathsf w$ (pA) adds the
current

```math
\mathsf w\,\alpha(t - t_s), \qquad \alpha(t) = \frac{e}{\tau_{\text{syn}}}\;t\;e^{-t/\tau_{\text{syn}}} \quad (t \ge 0) .
```

$\alpha$ rises to 1 at $t = \tau_{\text{syn}}$ and decays; $\mathsf w$ is the peak of the
pulse. Its charge is

```math
\int_0^\infty \mathsf w\,\frac{e}{\tau}\,t\,e^{-t/\tau}\,dt
= \mathsf w\,\frac{e}{\tau}\,\Big[\tau^2\Big]
= \mathsf w\,e\,\tau_{\text{syn}} ,
```

using $\int_0^\infty t\,e^{-t/\tau}dt = \tau^2$. A sender firing at the rate $r$ therefore
delivers the mean current

```math
\bar I = r\,\mathsf w\,e\,\tau_{\text{syn}} .
```

This one line turns every weight of the three repos into a current, and back: to deliver a
mean current $\bar I$ per unit of rate one needs $\mathsf w = \bar I/(e\,\tau_{\text{syn}})$.

*How smooth the current is.* A pulse lasts about $e\,\tau_{\text{syn}}$ (its area over its
peak). Senders that fire $\Lambda$ spikes a second in all make $\Lambda\,e\,\tau_{\text{syn}}$
pulses overlap at any moment. Few overlapping pulses show as ripple.

*As a filter.* The Laplace transform of $\alpha$ is $e\,\tau/(1 + s\tau)^2$: an alpha synapse
is two first-order lags of $\tau_{\text{syn}}$ in a row. This is used in D.12.

**In spikes.** A positive weight uses the excitatory time constant, a negative one the
inhibitory. Where a signal has both signs and must rise and fall as one (the sheet's
anchor, D.4), a synapse type that carries both signs with one time constant is used.

**Built.** NEST's `iaf_psc_alpha` and `iaf_psc_alpha_multisynapse`.

**Measured.** The encoder of a ring: 200 Hz through a weight of 4,500 pA and 27 ms gives
$200\cdot4500\cdot e\cdot0.027 = 66{,}054$ pA, 0.147 of the rheobase (A.1). A slow synapse
is chosen where spikes are few: a ring's bump sends about 1,020 spikes a second, so with
15 ms about 40 pulses overlap, with 5 ms it would be 14.

**To derive further.** The ripple of the summed current as a function of $\Lambda$ and
$\tau_{\text{syn}}$, and what it does to the receiving neuron's rate.

## A.3 Three kinds of signal

| Signal | Where | A value is | Zero is |
| --- | --- | --- | --- |
| place code on a ring | joint rings, goal rings, output rings | which 27 or so of 200 neurons fire | – (silence means "no value") |
| place code on a sheet | forward sheets, goal sheets | which 40 or so of 2,304 neurons fire | – |
| weighted sum of rates | every population between them | $\sum_j \delta_j r_j$ | a firing pattern, not silence |

- **In a place code the neurons that fire are few and the rest are silent.** The value is
  where the activity is. Silence itself is a signal.
- **In a population about half of the neurons fire whatever the value is.** A population
  that holds 0 fires: its positive and negative decoders cancel. A sum population of 60
  neurons at the value 0 sends 93 spikes per tick from 31 neurons.
- **Crossing between the two** happens four times in the full pose network: ring to cos and
  sin (C.3), numbers to a packet (D.4), a packet to numbers (D.11), and a share to the lead
  of a goal ring (C.10).
- **The spikes are regular, not random.** A neuron under a nearly constant current fires
  like a clock: the gaps between the spikes of a population neuron vary by 2% to 7% of
  their mean (a Poisson train: 100%). The neurons start from different potentials, so the
  summed current is smooth. The only random numbers are those of the encoders' spike
  generators.

## A.4 A population that holds a value

**What it is for.** To hold a number or a pair (a cosine, a matrix entry, an error) as
firing rates, so that it can be added to others, multiplied, and read by the next stage.

**Mathematics.** A population of $n_p$ neurons holds a value $x$ (a number, or a pair). Each
neuron has an *encoder* $\epsilon_i$, the direction of $x$ it responds to, and receives the
current

```math
J_i(x) = \alpha_i\,\frac{\epsilon_i \cdot x}{\varrho} + \beta_i .
```

Gain $\alpha_i$ and bias $\beta_i$ are fixed by two conditions on the projection
$\xi = \epsilon_i\cdot x/\varrho$:

- the neuron starts to fire at its own *intercept* $\xi^0_i$: $J_i = I_{th}$ at $\xi = \xi^0_i$;
- it reaches its own *maximum rate* $f^{\max}_i$ at the edge of the range: $J_i = J^{\max}_i = J(f^{\max}_i)$ at $\xi = 1$.

Two linear equations in $\alpha_i, \beta_i$:

```math
\alpha_i\,\xi^0_i + \beta_i = I_{th}, \qquad \alpha_i + \beta_i = J^{\max}_i
\;\;\Longrightarrow\;\;
\alpha_i = \frac{J^{\max}_i - I_{th}}{1 - \xi^0_i}, \qquad \beta_i = I_{th} - \alpha_i\,\xi^0_i .
```

Each neuron is then a ramp in $\xi$ that starts at $\xi^0_i$; the population is many ramps
with different starts, slopes and directions.

**In spikes.** The bias $\beta_i$ is a constant current of the neuron itself. The term
$\alpha_i\,\epsilon_i\cdot x/\varrho$ never exists as a number: it arrives as the sum of the
pulses of all incoming spikes (A.5).

**Built** (SO-3 `ring_so3/neurons.py`). Intercepts uniform in $[-0.9, 0.9]$, maximum rates
uniform in $[100, 200]$ Hz, drawn from a seed.

| Kind | Holds | Encoders | Neurons |
| --- | --- | --- | ---: |
| scalar ("sum") | one number in $[-1, 1]$ | $+1$ or $-1$, half each | 60 |
| pair | $(\cos, \sin)$ or another pair | all round the circle | 120 |
| product | a pair $(a, b)$ | the four diagonals $(\pm1, \pm1)/2$ | 200 |

**Measured.** One neuron of a sum population with intercept 0.37, $\alpha = 642$ pA,
$\beta = -52$ pA, while the value steps from 0 to 0.7: the current settles at 399 pA
(formula: $642\cdot0.7 - 52 = 397$ pA) with a ripple of ±14 pA; first spike 27 ms after the
step; then 68 Hz.

**To derive further.** How many neurons a given accuracy needs (the error of a readout
against $n_p$).

## A.5 Reading a value and passing it on

**What it is for.** To get a value, or a function of it, out of a population's rates, and to
hand it to the next population as synaptic weights.

**Mathematics: the readout.** A function $y(x)$ is read as $\hat y = \sum_i \delta_i\,r_i(x)$.
The *decoders* $\delta$ minimise the squared error over samples $x^{(1)},\dots,x^{(S)}$ with
a penalty. With $A_{si} = r_i(x^{(s)})$ and $Y_s = y(x^{(s)})$,

```math
\delta = \arg\min_\delta\; \lVert A\delta - Y\rVert^2 + S\sigma^2\lVert\delta\rVert^2 .
```

Setting the gradient $2A^\top(A\delta - Y) + 2S\sigma^2\delta$ to zero:

```math
\delta = \left(A^\top A + S\sigma^2 I\right)^{-1} A^\top Y, \qquad \sigma = 0.02\,\max A .
```

*Why the penalty has this form.* Let every rate carry independent noise $\eta_i$ of variance
$\sigma^2$. The read value becomes $\sum_i \delta_i (r_i + \eta_i)$; its expected squared
error gains $\mathbb E[(\sum_i \delta_i\eta_i)^2] = \sigma^2\lVert\delta\rVert^2$ per sample,
$S\sigma^2\lVert\delta\rVert^2$ over $S$ samples. The fit then does not rely on differences
of rates that are smaller than the noise.

*Why a sum of ramps can give a function.* Each neuron contributes a ramp along its encoder
that starts at its own intercept. A weighted sum of ramps with different starts is a
piecewise-smooth curve that can follow any smooth function of $\epsilon\cdot x$; with
encoders in several directions, sums of such functions of different projections are
available. The same rates can be read with several sets of decoders: the value, its square
(D.4), a bounded version of it (C.9), a ratio (D.11).

**Mathematics: the connection.** To hand $\hat y$ decoded from population A to population B
as the coordinate $t$ of B's value, neuron $i$ of B must receive the mean current
$\alpha_i(\epsilon_i\cdot t)\,\hat y/\varrho$. A spike of sender $j$ through the weight
$\mathsf w_{ij}$ delivers the mean current $\mathsf w_{ij}\,r_j\,e\,\tau_{\text{syn}}$ (A.2).
Equating term by term in $r_j$:

```math
\sum_j \mathsf w_{ij}\,r_j\,e\,\tau_{\text{syn}} = \frac{\alpha_i(\epsilon_i\cdot t)}{\varrho}\sum_j \delta_j r_j
\;\;\Longrightarrow\;\;
\mathsf w_{ij} = \frac{\alpha_i\,(\epsilon_i\cdot t)}{\varrho}\cdot\frac{\delta_j}{e\,\tau_{\text{syn}}} .
```

The weight matrix is the outer product of B's input weights and A's decoders: rank one per
value passed. Inputs from several populations add in the membrane, which is how a **sum**
is computed, and a constant term is a constant current added to the bias.

**In spikes.** Step 3 of one population (spikes to current) *is* step 1 of the next (value
to current). There is no place where $\hat y$ is stored and multiplied by $\alpha_i\epsilon_i$.
Between two populations a weight is a few pA (median size 4 to 7 pA, largest 140 to 250 pA,
about half negative), against a rheobase of 187.5 pA: one spike decides nothing; a value is
carried by the thousands of spikes a second that reach every neuron.

**Built.** SO-3 `neurons.solve_decoders`, `ring_so3/nest/calibrate.py`,
`ring_so3/nest/populations.py`. The decoders used in the networks are fitted to rates
*measured in NEST* with the populations driven by spikes; the closed form $f(J)$ is used only
for copies without spikes. No pose of any arm enters a fit, so one calibration serves every
arm.

**Measured.** A readout cell (a neuron that never fires, $\tau = 50$ ms, weights
$\delta_j\cdot50\ \text{pA}/(e\,\tau_{\text{syn}})$) settles at 10 mV per unit of value: this
is how tests and pages read a value. Nothing in the network listens to it.

**To derive further.** The error of a decoded value as a function of the number of neurons
and of $\sigma$; why fitting on measured rates beats fitting on $f(J)$ (the pulse train is
not a constant current).

## A.6 A population that multiplies

**What it is for.** The only non-linear operation between rings and sheets.

**Mathematics.** The identity

```math
ab = \tfrac14\left[(a+b)^2 - (a-b)^2\right]
```

makes a product of two numbers the difference of two functions of *one* number each. A
population holds the pair $(a, b)$ with encoders on the four diagonals
$\epsilon \in \{(\pm1,\pm1)/2\}$. A neuron then responds to
$\epsilon\cdot(a,b) = \pm(a+b)/2$ or $\pm(a-b)/2$: it sees only one of the two sums. By A.5
a weighted sum of neurons that all see $a+b$ can give $(a+b)^2$, and likewise for $a-b$; one
set of decoders, fitted to $y = ab$, gives the difference.

*Why diagonal encoders.* A neuron with a general encoder $(\cos\gamma, \sin\gamma)$ sees
$a\cos\gamma + b\sin\gamma$. Only the diagonal directions enter the identity; putting all
neurons there spends none on directions that do not help.

**In spikes.** The 200 neurons are four groups of 50. The spikes that carry $a$ and those
that carry $b$ reach all of them, with weights of such signs that a neuron of the first
group gets $\alpha_i(a+b)/2 + \beta_i$, of the second $\alpha_i(a-b)/2 + \beta_i$, and so on.
No neuron sees $a$ or $b$ alone, and the product is nowhere in the spikes as a rate: it is
$\sum_i\delta_i r_i$. As the identity says, the fitted decoders are positive in the two
groups that see $\pm(a+b)$ and negative in the two that see $\pm(a-b)$, and within a group
they are largest for the neurons that start to fire late, which is how a sum of ramps
becomes a square.

**Built.** SO-3 `ring_so3/neurons.py` (kind `product`), decoders fitted on a 25 x 25 grid of
measured rates.

**Measured.** A product is right to 0.0077 (root mean square) on inputs it was not fitted
on. 110 of 200 neurons fire, at 82 Hz, 447 spikes per tick. After one multiplying stage a
vector's length is off by up to 4.7% (a mark of 3% was missed).

**To derive further.** The decoders in closed form for an idealised ramp neuron (a square
as an integral of ramps); the error of the product as a function of the neurons per group.

## A.7 What a derivation becomes in spikes

Every derivation in parts B to D ends in one of five forms. This table says what each form
is in the network; the "In spikes" item of every part applies it.

| In the derivation | In the network |
| --- | --- |
| a value $x$ that is used later | a population (60, 120 or 200 neurons) whose rates hold it; never a single neuron |
| a constant times a value, $c\,x$ | nothing of its own: $c$ is a factor on the weights into the next population |
| a sum $x + y$ | the spikes of both arrive at one population; currents add in the membrane |
| a product $x\,y$ | one product population of 200 neurons (A.6) |
| a function of one value, $\varphi(x)$ | another set of decoders on the population that holds $x$ |
| a division | not available. Avoided by a constant divisor (D.9) or by a readout fitted to a ratio over a narrow range (D.11) |
| a place on a ring or sheet | a bump or packet: which neurons fire |
| "the angle of a pair $(x,y)$" | a ring driven by $x\cos\theta_j + y\sin\theta_j$ above a threshold (C.4) |
| a time derivative or integral | not computed. Motion comes from a share that leads the state (C.10); integration only as a packet pushed over a sheet (D.5) |
| a clip or a bound | a readout fitted to stay at ±1 outside its range (C.9) |

Three things a derivation does not contain and the network adds:

- **Delay.** Every stage is a filter. A bump follows a moving angle about 0.09 s behind; a
  5 ms synapse and a 20 ms membrane add a few milliseconds per population; a readout over
  ticks adds its window. The loop equations of B.10, C.11 and D.14 carry the sum as $\tau$.
- **Ripple.** A value decoded from spikes fluctuates; the fewer the spikes, the more. A
  packet of 25 spikes per tick is the extreme case (D.11).
- **Bounds.** A population holds numbers in $[-1,1]$, so every quantity is scaled by a bound
  that must be proved (D.1), and what exceeds it is held at the bound.

---

# Part B. SO(2): one ring per joint

Repo: `Ring-Attractor-SO-2-Joint-Architecture` (package `tiago_ring_controller`). Its
reference: that repo's `TECHNICAL.md`. File names in this part are relative to
`src/tiago_ring_controller/` (scripts) and `src/tiago_ring_controller/src/tiago_ring_controller/`
(modules) of that repo.

## B.1 The task; joint angle and ring index

**What it is for.** One revolute joint has an angle $q \in [q_{\min}, q_{\max}]$: a rotation
about one axis, an element of SO(2), a point on a circle. A ring of neurons holds it as the
position of a bump. The controller has to take the joint to a goal $q^\ast$.

**Mathematics.** The joint range lies on an arc that covers the fraction
$f_{\text{arc}} = 0.7$ of the ring, centred on index $N/2$:

```math
i(q) = i_0 + \frac{q - q_{\min}}{q_{\max} - q_{\min}}\,S, \qquad S = f_{\text{arc}}N = 140, \qquad i_0 = \frac{N - S}{2} = 30 ,
```

```math
\rho_q = \frac{q_{\max} - q_{\min}}{S}\ \ \text{rad per index}, \qquad q(i) = q_{\min} + (i - i_0)\,\rho_q .
```

A joint without limits (the generic chains of parts C and D) uses the whole ring,
$\rho_q = 2\pi/N$, and the ring is circular.

**In spikes.** The 60 indices that are left lie around the seam (index 0 = index $N$). Two
things keep a bump out of them:

- **the wall**: neurons outside the indices 15 to 185 get the bias $0.9\,I_{th}$; by A.1
  they are 10% of the rheobase below threshold and no bump can rest there;
- **fading at the limits**: the gain population that moves the bump toward a limit is driven
  less and less over the last 10 indices before it (B.7).

**Built.** SO-2 `control/mapping.py`, `math/control.py`. An older mapping put the range on
the indices 5 to 195, so that the two limits were neighbours across the seam and a bump near
one could slip to the other; the bounded mapping exists because of that.

**Measured.** $\rho_q$ = 0.020 rad for arm_6, 0.034 rad for the joints with the widest range.

## B.2 The ring attractor

**What it is for.** To hold one number as the place of a bump, without input, at any place.

**Mathematics.** Every pair of ring neurons is connected with a weight that depends only on
their distance along the ring. Distances are scaled so that the far side is at $d = 50$ (one
index is 0.5). With $\sigma_1 = 10$, $\sigma_2 = 5$,

```math
w(d) = 3\;\frac{\sigma_2\,e^{-d^2/2\sigma_1^2} - \sigma_1\,e^{-d^2/2\sigma_2^2}}{\sigma_2 - \sigma_1}
     = 3\left(2\,e^{-d^2/50} - e^{-d^2/200}\right) :
```

a narrow excitation minus a wide inhibition.

*Zero crossing.* $w(d) = 0$ where $2e^{-d^2/50} = e^{-d^2/200}$. Taking logarithms,
$\ln 2 = d^2\left(\tfrac1{50} - \tfrac1{200}\right) = \tfrac{3d^2}{200}$, so

```math
d_0 = \sqrt{\tfrac{200}{3}\ln 2} = 6.8 \quad (13.6\ \text{indices}) .
```

*Strongest inhibition.* $w'(d) = 0$: $-\tfrac{4d}{50}e^{-d^2/50} + \tfrac{2d}{200}e^{-d^2/200} = 0$,
so $e^{-3d^2/200} = \tfrac18$, $d = \sqrt{\tfrac{200}{3}\ln 8} \approx 11.8$ (about 24
indices), where $w \approx -1.13$.

*Far away the weight is zero*: $w(30) \approx -0.03$. Two bumps more than about 60 indices
apart do not feel each other.

**Model: why a bump forms and stays.** Neurons inside a group excite each other
($w > 0$ for $d < d_0$). Every neuron rests at threshold (A.1), so that excitation alone
keeps them firing. Just outside the group the net input is negative (the inhibition of all
bump neurons at distances 7 to 30), and a neuron at threshold with a negative input is
silent. The width settles where a neuron at the edge receives as much excitation as
inhibition.

**Model: why the bump can be anywhere.** $w$ depends only on the distance, so the network
is unchanged by a rotation of the ring. If a bump at one place is a stable state, so is the
same bump at every place: a circle of stable states, a continuous attractor. This has two
sides: a small asymmetric input moves the bump and it stays where it was moved (B.7); and
noise moves it too.

**Model: why a bump does not jump.** A stimulus far from the bump lies where $w \approx 0$:
it neither feels the bump nor is felt by it. It raises a second bump instead of moving the
first. So an angle is only ever changed in small steps (B.3), and a goal is made to travel
(C.11, D.14).

**In spikes.** The current that the ring's own spikes send to a neuron at each distance
from the centre of a held bump, from the measured rates and $w(d)$ (SO-3 census):

| Distance from the centre (neurons) | 0 | 5 | 6 | 8 | 10 | 13 | 14 | 20 | 30 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Current from the ring's own spikes (pA) | +174 | +142 | +130 | +100 | +65 | +9 | −10 | −100 | −116 |
| Rate that current alone gives (Hz) | 18 | 18 | 17 | 17 | 16 | 13 | 0 | 0 | 0 |
| Rate measured (Hz) | 66 | 67 | 17 | 17 | 16 | 13 | 0 | 0 | 0 |

- One recurrent spike carries 0.22 pC, 1,500 times less than a generator spike. It matters
  all the same because the receiver sits at threshold: an extra 100 pA is
  $\varepsilon \approx 2\cdot10^{-4}$, and $1/(7.04\ \text{ms}\cdot\ln(1/\varepsilon)) \approx 17$ Hz.
- The 11 stimulated neurons fire at 66 to 68 Hz (the generators decide); 8 neighbours on
  each side at 13 to 18 Hz from the ring's own spikes alone; from the 14th neuron on the
  current is negative and the bump ends. Together 27 neurons, about 1,020 spikes a second.

**Built.** SO-2 `math/ring.py`, block `Ring`; SO-3 and SE-3 reuse it (`ring_so3/nest/ring.py`).

**Measured.** About 28 active neurons in a free bump, about 35 Hz each. Drift of a free
bump: about 2 indices in 20 s.

**To derive further.** A rate model of the ring (a neural field with this kernel and the
threshold-linear, logarithmic gain of A.1): existence, width and stability of the bump, and
the drift as a diffusion driven by spike noise. Everything above is argument plus
measurement.

## B.3 Writing an angle

**What it is for.** To put a bump at the place of a given angle.

**Mathematics.** One spike generator per ring neuron. To write $q$, the generators of the 11
neurons $i(q)-5,\dots,i(q)+5$ fire at 200 Hz with the weight 4,500 pA. By A.2,

```math
\bar I = 200\ \text{Hz}\cdot4500\ \text{pA}\cdot e\cdot0.027\ \text{s} = 6.6\cdot10^4\ \text{pA} = 0.147\,I_{th},
\qquad f = \frac{1}{7.04\ \text{ms}\cdot\ln(1.147/0.147)} \approx 69\ \text{Hz} .
```

*Between two indices.* Writing to the nearest index holds the angle to half an index, 0.9°
on a circular ring of 200. With $x = \lfloor x\rfloor + f_x$ the position on the ring, the
stimulus is a blend of two windows,

```math
\text{rate}_i = (1 - f_x)\,W_{\lfloor x\rfloor}(i) + f_x\,W_{\lfloor x\rfloor+1}(i), \qquad W_c(i) = 200\ \text{Hz for } |i - c| \le 5,\ 0\ \text{otherwise} .
```

The nine neurons in the middle get the full rate, the two at the ends $(1-f_x)$ and $f_x$ of
it. The centre of the stimulus is $\lfloor x\rfloor(1-f_x) + (\lfloor x\rfloor+1)f_x = x$.

**In spikes.** The generators are random (Poisson); the ring neurons are not: each
integrates about three generator spikes per spike of its own over a large capacitance, and
the gaps between its spikes vary by 2%.

*When the encoder stimulates* (SO-2): `once` (the first tick), `continuous` (every tick), or
`corrective` (only while the bump is more than a dead band from $i(q)$). A stimulated bump
fires about twice as hard as a free one, and a comparator that reads rates takes a harder
bump for a bump elsewhere (B.5). Stimulating only when the bump is off keeps a resting bump
at its free strength and still cancels drift. SO-3 and SE-3 write every tick, fractionally,
and their readouts are fitted on bumps written that way (C.3).

**Built.** SO-2 block `Encoder`; SO-3 `nest/ring.py: fractional_bump_rates`.

**Measured.** With fractional writing a seven-joint pose is held to 0.29°, with
nearest-index writing to 0.96°.

## B.4 Reading an angle: the population vector

**What it is for.** To get the bump's place as a number, for tests, pages and the robot.

**Mathematics.** From the spike counts $c_i$ of the ring neurons over the last ticks:

```math
z = \sum_i c_i\,e^{\mathrm i\theta_i}, \qquad \hat\imath = \frac{N}{2\pi}\arg z \pmod N, \qquad \text{strength} = \frac{|z|}{\sum_i c_i} .
```

*Why this is the centre.* Let the counts be symmetric about $\varphi$:
$c(\theta) = b(\theta - \varphi)$ with $b$ even. Substituting $u = \theta - \varphi$,

```math
z = e^{\mathrm i\varphi}\sum_u b(u)\,e^{\mathrm iu} = e^{\mathrm i\varphi}\Big[\sum_u b(u)\cos u + \mathrm i\underbrace{\sum_u b(u)\sin u}_{=0}\Big] ,
```

so $\arg z = \varphi$ exactly. The strength is 1 for a tight bump and near 0 for activity
spread round the ring.

**In spikes.** This readout is not made of neurons: spike counts are taken from recorders.
It is the one place where a number leaves the network.

**Built.** SO-2 block `RingState`, `math/circular.py`; SO-3 `ring_so3/mapping.py`.

**To derive further.** The variance of $\hat\imath$ for a bump of $n_s$ spikes in the window
(a circular mean of $n_s$ samples from the bump's profile).

## B.5 The Fourier readout

**What it is for.** To turn the place of a bump into a few rates that the comparator can
weigh.

**Mathematics.** $2K = 40$ feature neurons read a ring. For each harmonic $k = 1,\dots,K$
there is a pair with the weights from ring neuron $i$

```math
m^+_{k,i} = \max(\sin k\theta_i, 0), \qquad m^-_{k,i} = \max(-\sin k\theta_i, 0) .
```

Weights of one sign can be made by excitatory synapses; the signed feature is the
difference of the pair:

```math
s_k = \sum_i (m^+_{k,i} - m^-_{k,i})\,r_i = \sum_i \sin(k\theta_i)\,r_i .
```

*What the features say about the bump.* Let $r_i = b(\theta_i - \varphi)$ with $b$ even.
With $u = \theta_i - \varphi$ and $\sin k(u+\varphi) = \sin k\varphi\cos ku + \cos k\varphi\sin ku$,

```math
s_k = \sin k\varphi\underbrace{\sum_u b(u)\cos ku}_{B_k} + \cos k\varphi\underbrace{\sum_u b(u)\sin ku}_{=0} = B_k\sin k\varphi .
```

The features are the sine series of the bump's place, each scaled by the Fourier coefficient
$B_k$ of the bump's shape.

- A wider bump has smaller $B_k$ at high $k$: this limits how sharp a function of the place
  the features can express.
- **Amplitude sensitivity.** A stronger bump scales every $B_k$, hence every $s_k$. Any
  weighted sum of the features grows with the bump's strength as well as with its place.
  This is the weakness that the corrective encoder works around here, and that level cells
  remove on the sheets (D.12).

**In spikes.** The feature neurons are default `iaf_psc_alpha` cells with a 200 pA bias;
they turn their input into a rate through their own threshold, so their counts are a
monotone, not exactly linear, function of the sums. The fit of B.6 is made on measured
counts and absorbs that.

**Built.** SO-2 block `FourierReadout`, `math/fourier.py` (scale 200 on the weights).

## B.6 The ramp comparator

**What it is for.** To say on which side of the reference bump the moved bump is.

**Mathematics. Step 1: a number that rises along the ring.** A weight vector $\nu$ on the
signed features gives a function of the bump's place,

```math
\Gamma(\varphi) = \sum_{k=1}^K \nu_k\,s_k(\varphi) \approx \sum_k \nu_k B_k \sin k\varphi .
```

The target is a ramp that rises from 0 to 1 across the arc and falls back across the seam
(rising over the fraction 0.8 of the ring). After its mean is removed it is odd about the
seam, so it has a sine series; that is why sine features suffice. The weights are a ridge
fit to *measured* feature counts $\mathsf S$ (one row per bump position):

```math
\nu = \arg\min_\nu \lVert \mathsf S\nu - \Gamma\rVert^2 + \lambda\lVert\nu\rVert^2 = (\mathsf S^\top\mathsf S + \lambda I)^{-1}\mathsf S^\top\Gamma .
```

**Step 2: two ramps and a decision.** Two neurons carry the ramp: *warm* is driven by the
features of the ring that is moved, *cold* by those of the reference ring, each with the
weights $\nu$. Two decision neurons compare them:

```math
\text{cold}\xrightarrow{+}\text{right},\quad \text{cold}\xrightarrow{-}\text{left},\quad
\text{warm}\xrightarrow{+}\text{left},\quad \text{warm}\xrightarrow{-}\text{right},\quad
\text{left}\overset{-}{\longleftrightarrow}\text{right} .
```

If the reference is further along the ring than the moved bump, cold fires more than warm,
*right* wins and the bump is moved up. With mutual inhibition only one side fires.

**Model: the blind band.** The decision is driven by
$\Gamma(\varphi_{\text{ref}}) - \Gamma(\varphi) \approx \Gamma'(\varphi)\,\Delta\varphi$. The
ramp rises by 1 over 160 indices, so an offset of 5 indices is a difference of 3% of its
range: below the spike-count noise of the two ramp neurons and the threshold of the decision
neurons. The comparator sees the direction from far away and is blind nearby.

**In spikes.** Each ramp is one neuron: its *rate* is the number. That is a third way of
carrying a value besides a place code and a weighted sum, and it is the reason for the
blind band: one neuron's spike count in a tick is a coarse number.

**Built.** SO-2 blocks `FourierReadout` + `Homeostasis`, `training/comparator.py`,
`scripts/fit_comparator.py`. The block is called `Homeostasis` in the code; it is this
comparator.

**Measured.** No drive in a band 10 to 21 indices wide around the reference.

**To derive further.** The width of the blind band from the Poisson-like count noise of the
two ramp neurons and the decision threshold. In SE(3) the same ramp is used without the
decision: the difference of two ramps is kept as a graded number (D.11).

## B.7 The gain pair: moving a bump

**What it is for.** To move a bump one way or the other by a signal, and to produce the
spikes that command the joint.

**Mathematics.** Two populations of $N$ neurons, *left* and *right*.

*An AND of two inputs.* Gain neuron $i$ of each side gets the spikes of ring neuron $i$
(one-to-one), the spikes of its decision neuron, and inhibition from the other decision
neuron. Its threshold is such that neither input alone makes it fire. So the *right* side
fires only where the bump is **and** the decision is "right": the gain populations carry a
copy of the bump, on the side that won.

*The copy moves the bump.* Each gain neuron inhibits the ring neuron next to its own:

```math
\text{right}_i \xrightarrow{\;w_g\;} \text{ring}_{i-1}, \qquad \text{left}_i \xrightarrow{\;w_g\;} \text{ring}_{i+1}, \qquad w_g = -0.6 .
```

With *right* active every bump neuron inhibits its lower neighbour. Inside the bump this is
a small uniform reduction. At the lower edge it decides: the edge neuron sits at the balance
of excitation and inhibition (B.2), and the extra inhibition silences it. At the upper edge
nothing is taken away. The bump loses neurons at the bottom and, keeping its width, gains
them at the top: it moves up.

**Model: how fast.** With $n_R$, $n_L$ the spikes of the two sides in a tick and
$s = n_R - n_L$, the displacement per tick is proportional to $s$ above a dead zone:

```math
\Delta i = \mu_b\,\operatorname{dz}_{s_0}(s), \qquad \operatorname{dz}_{s_0}(s) = \operatorname{sign}(s)\max(|s| - s_0, 0) .
```

Below $s_0$ the push is too weak to silence the edge neuron and the bump stays, pinned to
the lattice of neurons.

**In spikes.** The gain neuron has no bias (SO-2) and a threshold of 1.35 mV; ring spikes
arrive with the weight $4.5\cdot10^5$, decision spikes with $10^4$ and $-3\cdot10^4$. In SO-3
and SE-3 the gain pair is used differently: it does not move an attractor's bump but
*writes* a goal ring beside the state bump (C.10).

**Built.** SO-2 block `Gain`, `nest/populations.py`; fit `scripts/calibrate_transport.py`.

**Measured** (96 trials on NEST): $\mu_b = 0.005332$ indices per spike, $s_0 = 20$ spikes
per tick. A linear model without dead zone predicts the displacement to 10.4% in the median
on held-out trials, the dead-zone model to 4.8%.

**To derive further.** The speed of a bump under asymmetric inhibition from the ring
dynamics (how many edge spikes must be removed to shift the edge by one neuron), which would
give $\mu_b$ and $s_0$ from the ring's parameters.

## B.8 The decoder: spikes to joint velocity

**What it is for.** To turn the gain pair's spikes into a command for the joint.

**Mathematics.** Per tick of length $dt$:

```math
x_t = \operatorname{dz}_{s_0}(n_{R,t} - n_{L,t}), \qquad
y_t = \lambda_s\,y_{t-1} + (1 - \lambda_s)\,x_t, \quad \lambda_s = e^{-dt/\tau_s}, \qquad
v_t = k\,y_t ,
```

clipped to the joint's velocity limit: a first-order low-pass with time constant $\tau_s$
and a gain. The arm takes positions, so the velocity is integrated,
$q^{\text{cmd}}_{t+1} = \operatorname{clip}(q^{\text{cmd}}_t + v_t\,dt)$, and sent as a short
trajectory.

*The gain as a calibration (v3).* If the bump is the model's belief about the joint and the
same spikes move both, the joint must move as far as the bump:

```math
\Delta q = \rho_q\,\Delta i \;\Longrightarrow\; v\,dt = \rho_q\,\mu_b\operatorname{dz}_{s_0}(s) \;\Longrightarrow\; k = \frac{\mu_b\,\rho_q}{dt} .
```

$\mu_b$ and $s_0$ belong to the network and $\rho_q$ to the joint, so one fit serves all
joints.

**In spikes.** Outside the network: spike counts in, a number out. This is the output of
the SO(2) controller. In SO-3 and SE-3 the output is instead the place of a goal ring's
activity (C.10).

**Built.** SO-2 block `Decoder`, `math/control.py`, `graph/calibration.py`.

**Measured.** In Gazebo the joint follows the streamed position to within 2% with a lag of
5 to 14 ms.

## B.9 Two rings, open loop (v3)

**What it is for.** The simplest controller: a goal ring G and a state ring S.

**Mathematics.** The encoder writes $i(q^\ast)$ on G and $i(q_0)$ on S, once. The comparator
compares S (warm) with G (cold); the winning gain side moves S toward G; the decoder turns
the same spikes into joint velocity. With $\pi$ the index of the state bump,

```math
\pi_{t+1} = \pi_t + \mu_b\operatorname{dz}_{s_0}(s_t), \qquad q_{t+1} = q_t + \gamma\,k\,dt\;\tilde s_t ,
```

where $\tilde s$ is the filtered $\operatorname{dz}_{s_0}(s)$ and $\gamma$ is how much the
joint really moves per commanded unit. Summing over a trial with $k = \mu_b\rho_q/dt$ (the
filter's total equals its input's total):

```math
q_{\text{end}} - q_0 = \gamma\,\rho_q\,(\pi_{\text{end}} - \pi_0) .
```

The joint is never looked at after the start, so every error stays in the result:

```math
q_{\text{end}} - q^\ast = \underbrace{\rho_q(\pi_{\text{end}} - i(q^\ast))}_{\text{bump stops off the goal}} + \underbrace{(\gamma - 1)\rho_q(\pi_{\text{end}} - \pi_0)}_{\text{actuator gain}} + \text{rounding, drift, disturbances} .
```

**In spikes.** One comparator, one gain pair; the trial ends when the bumps are within the
blind band and the comparator falls silent.

**Built.** SO-2 `graph/templates.py: two_ring_single_joint`.

**Measured.** The bump stops a median of 3.1 indices from the goal (the blind band), the
joint a median of 1.5 indices from the bump. Over all seven joints in Gazebo v3 ends
0.091 / 0.257 / 0.344 rad from the goal (median / 90th percentile / largest) and settles in
54% of the trials. A motor gain 30% low leaves the joint 30% of the move short, as the
second term says.

## B.10 Three rings, closed loop (v4)

**What it is for.** To make the result independent of the actuator: the body's own angle is
in the loop.

**Mathematics.**

| Ring | Holds | Written by |
| --- | --- | --- |
| G, goal | $i(q^\ast)$ | the goal encoder |
| S, state | where the joint is, $i(q)$ | the joint's encoder, all the time |
| P, predictive | where the joint is expected to be | the measured angle once, then only two gain pairs |

Two comparisons act on P, each with its own gain pair: the **goal pair** (P against G) moves
P toward the goal and its spikes are the motor signal; the **body pair** (P against S) moves
P toward the body and commands nothing.

*The projection: a comparison that works nearby.* The reference ring projects straight onto
the gain pair. Gain neuron $i$ of the right side receives

```math
I^R_i = w_e\sum_{j=i+1}^{i+W} r^{\text{ref}}_j - w_h\sum_{j=i-W}^{i-1} r^{\text{ref}}_j, \qquad W = 10 ,
```

and the left side the mirror image. Take both bumps as blocks of the same width and rate
$r$, the reference shifted up by $\Delta$ indices ($0 < \Delta < W$). For a gain neuron in
the middle of the P bump the window above holds $\Delta$ more reference neurons than a
centred window, the window below $\Delta$ fewer. With $n_0$ the reference neurons in a
window when the bumps coincide,

```math
I^R_i \approx r\big[(w_e - w_h)\,n_0 + (w_e + w_h)\,\Delta\big], \qquad I^L_i \approx r\big[(w_e - w_h)\,n_0 - (w_e + w_h)\,\Delta\big] .
```

The first term is the same on both sides and negative, since $w_h > w_e$: with the bumps on
top of each other neither side is driven. The second has opposite signs: the side that
leads toward the reference is driven in proportion to $\Delta$. The projection reaches only
$W$ indices; the ramp comparator covers the rest. Each comparison of v4 uses both.

*The loop.* Let $g = i(q^\ast)$, $\pi$ the predictive bump, $\sigma_s = i(q)$ the state bump,
and $u_g = \phi_g(g - \pi)$, $u_b = \phi_b(\sigma_s - \pi)$ the signed spike counts of the two
pairs, each a rising, odd function. Per tick:

```math
\pi_{t+1} = \pi_t + \mu_g u_g + \mu_{b'} u_b, \qquad q_{t+1} = q_t + \gamma\,k\,dt\;\widetilde{\operatorname{dz}}(u_g) .
```

*Where it comes to rest.* Both right-hand sides vanish:
$\widetilde{\operatorname{dz}}(u_g) = 0 \Rightarrow u_g \approx 0 \Rightarrow \pi \approx g$;
then $\mu_{b'}u_b = 0 \Rightarrow \sigma_s = \pi$. So $\sigma_s = \pi = g$: the body is at the
goal, to within the dead zones, and $\gamma$ does not appear. That is the difference from
v3.

*Why the body pair is the stronger.* Linearise near the goal: $u_g \approx c_g(g - \pi)$,
$u_b \approx c_b(\sigma_s - \pi)$; write $\varepsilon = g - \sigma_s$ (the body's error) and
$\eta = \pi - \sigma_s$ (prediction ahead of body), and
$\beta_m = \gamma k\,dt\,c_g/\rho_q$ (indices the joint moves per tick per index of goal
offset). From the two updates, using $g - \pi = \varepsilon - \eta$ and
$\sigma_s - \pi = -\eta$:

```math
\begin{aligned}
\varepsilon_{t+1} &= \varepsilon_t - \beta_m(\varepsilon_t - \eta_t) \\
\eta_{t+1} &= \eta_t + \mu_g c_g(\varepsilon_t - \eta_t) - \mu_{b'} c_b\,\eta_t - \beta_m(\varepsilon_t - \eta_t)
\end{aligned}
```

If the joint follows exactly as the bump is moved ($\beta_m = \mu_g c_g$), $\eta$ only
decays: $\eta_{t+1} = (1 - \mu_{b'}c_b)\eta_t$. If not, $\eta$ settles where the body pair's
pull balances the mismatch. Setting $\eta_{t+1} = \eta_t$ and neglecting $\eta$ beside
$\varepsilon$ in the mismatch term:

```math
\eta \approx \frac{\mu_g c_g - \beta_m}{\mu_{b'}c_b}\,\varepsilon ,
```

small when $\mu_{b'}c_b$ is large. The error then decays as
$\varepsilon_{t+1} \approx (1 - \beta_m(1 - \eta/\varepsilon))\,\varepsilon_t$: a first-order
approach whose *rate*, not whose end point, depends on $\gamma$.

*Why only the goal pair drives the robot.* If one signal moved both bump and joint, they
would always move together and a difference between them could never shrink.

*The dead zone.* Without a dead zone in the decoder the goal pair always produces a few
spikes and a few spikes always move the joint a little: the joint hunts about the goal.

**In spikes.** The feedback weights of the two pairs are $-0.6$ (goal pair) and $-3.0$ (body
pair), so the body pair displaces the bump about five times as far per spike. Projection
inhibition is 3 to 7.5 times its excitation.

**Built.** SO-2 `graph/templates.py: three_ring_single_joint`, `blocks/composites.py:
JointTriple`, `connect_gain_reference`.

**Measured** (seven arm joints in Gazebo, 56 trials per system, SO-2 `docs/results.md`):

| System | Goal error, median / p90 / largest (rad) | Settled |
| --- | --- | ---: |
| v3 | 0.091 / 0.257 / 0.344 | 54% |
| PID baseline | 0.027 / 0.048 / 0.054 | 100% |
| **v4** | 0.008 / 0.022 / 0.045 | 100% |

- Blind band of the projection: 0 to 1 index (ramp comparator: 10 to 21).
- Motor gain at 0.7 and 1.3 of its value: v4 ends 0.013 and 0.010 rad from the goal in the
  median, v3 0.314 and 0.203 rad.
- A robot driven from a gain pair on the state ring overshoots by 0.37 to 0.64 rad.
- Without the decoder's dead zone the joint hunts ±1 index with a period of about 8 s.
- v4 is exact to about half a ring index (0.4 in the median) and repeats a goal to
  0.0133 rad.

**To derive further.** The two-dimensional linear map $(\varepsilon, \eta)$ in full: its
eigenvalues and the condition on $\mu_{b'}c_b$, $\mu_g c_g$, $\beta_m$ for stability and for
no overshoot, with the one tick of delay in each direction (B.11) included.

## B.11 Time, the baseline and scoring

- **Ticks.** NEST and the robot advance in ticks of $dt = 50$ ms, locked together: the
  network runs one tick with the joint angle of the previous tick; the robot then gets the
  command from that tick's spikes. One tick of delay in each direction.
- **Time scale.** `time_scale` $= K_t$ runs $K_t$ NEST steps per robot tick (default 3). The
  network does spike for spike what it would do anyway; the robot gets $K_t$ times the
  spikes per tick and moves $K_t$ times as fast. The loop of B.10 is unchanged in units of
  NEST steps.
- **PID baseline.**
  $v_t = \operatorname{clip}\big(k_p e_t + k_i I_t + k_d (e_t - e_{t-1})/dt,\ \pm0.5\big)$ with
  $e_t = q^\ast - q_t$, $I_t = \operatorname{clip}(I_{t-1} + e_t\,dt)$, $k_p = 2.0$,
  $k_i = 0.1$, $k_d = 0.05$; the same gains for every joint, not tuned.
- **Scoring**, one function for every system: goal error $|q^\ast - q_{\text{final}}|$;
  settled if the error stays within 0.05 rad; settling time; time to 90% of the move;
  repeatability.

---

# Part C. SO(3): orientation from the rings

Repo: `Ring-Attractor-SO-3-Joint-Architecture` (package `ring_so3`; a copy is carried in the
SE-3 repo). Its reference: that repo's `TECHNICAL.md`, parts I to III and appendix A.

## C.1 The orientation is a sum of products

**What it is for.** To show that the orientation of a link can be computed with the two
operations a network has: adding and multiplying two values.

**Mathematics.** A chain of $n$ revolute joints. Joint $k$ turns about one coordinate axis
of its own frame; its frame sits in the previous link with a fixed rotation $A_k$:

```math
R_k = R_{k-1}\,A_k\,\operatorname{Rot}(\text{axis}_k, q_k), \qquad R_0 = I, \qquad R = R_n\,A_{\text{tool}} .
```

With $c = \cos q$, $s = \sin q$:

```math
\operatorname{Rot}(x,q)=\begin{pmatrix}1&0&0\\0&c&-s\\0&s&c\end{pmatrix},\quad
\operatorname{Rot}(y,q)=\begin{pmatrix}c&0&s\\0&1&0\\-s&0&c\end{pmatrix},\quad
\operatorname{Rot}(z,q)=\begin{pmatrix}c&-s&0\\s&c&0\\0&0&1\end{pmatrix}.
```

*Each entry is linear in each joint's features.* Every entry of
$\operatorname{Rot}(\text{axis}, q)$ is one of $0, \pm1, \pm\cos q, \pm\sin q$: a linear
combination of the three features $\varphi_k = (1, \cos q_k, \sin q_k)$. A matrix product
multiplies entries of *different* factors, never two entries of the same factor. So every
entry of $R$ is a sum of terms with exactly one feature from each joint:

```math
R_{ij} = \sum_{m_1,\dots,m_n\in\{0,1,2\}} T_{ij,\,m_1\dots m_n}\;\varphi_{1,m_1}\varphi_{2,m_2}\cdots\varphi_{n,m_n},
\qquad \operatorname{vec}(R) = T\,\Phi, \quad \Phi = \varphi_1\otimes\cdots\otimes\varphi_n ,
```

with $T$ a fixed $9\times3^n$ matrix that depends only on the axes and the $A_k$.

*Example*, $R = \operatorname{Rot}(x,q_1)\operatorname{Rot}(y,q_2)$:

```math
R = \begin{pmatrix} c_2 & 0 & s_2\\ s_1s_2 & c_1 & -s_1c_2\\ -c_1s_2 & s_1 & c_1c_2\end{pmatrix} .
```

*Most products never occur*, because a joint leaves one row and one column alone: for 2 to 7
joints of the chains here 8, 13, 21, 34, 55, 89 of the $3^n - 1$ products have a non-zero
column in $T$ (for 7 joints: 89 of 2,186).

*Cost of doing it all at once.* A product of $m$ features needs $m-1$ pairwise products;
sharing prefixes, the 89 products of 7 joints need 166 pairwise products.

**In spikes.** Not built this way for long arms; C.2 evaluates the same sum at a cost that
grows by a fixed amount per joint. A version that computes all at once exists for 2 and 3
joints and gives the same result.

**Built.** SO-3 `ring_so3/kinematics.py` builds $T$ and $\Phi$; a test confirms $T\Phi = R$
to $10^{-9}$.

## C.2 Joint by joint

**What it is for.** The same sum, evaluated so that every link's orientation comes out on
the way and the cost per joint is constant.

**Mathematics.** Write $M = R_{k-1}A_k$ (known before joint $k$) with columns
$M_x, M_y, M_z$. Multiplying by $\operatorname{Rot}(\text{axis}, q)$ on the right mixes the
two columns that are not the axis and leaves the axis column alone. For the axis $z$:

```math
R_k = M\operatorname{Rot}(z,q) = \big(M_x c + M_y s,\;\; -M_x s + M_y c,\;\; M_z\big) ,
```

and in general, with $u, v$ the two other axes in cyclic order,

```math
\text{col}'_u = \text{col}_u\cos q + \text{col}_v\sin q, \qquad \text{col}'_v = -\text{col}_u\sin q + \text{col}_v\cos q, \qquad \text{col}'_{\text{axis}} = \text{col}_{\text{axis}} .
```

- **Cost.** Each of the three rows needs four products (two entries times cos and sin): 12
  per joint. $A_k$ only re-labels and re-signs columns and needs no neurons. A product with
  a constant factor is no product, so the first joints are cheaper: 4, 14, 26, 38, 50, 62
  pairwise products for 2 to 7 joints, 12 more for each further joint.
- **It is the same sum.** Expanding the recursion gives exactly the terms of $T\Phi$.
- **The joint's axis in the base frame** is the column that is left alone,
  $a_k = (R_k)_{\cdot,\text{axis}_k}$. Everything about inverse kinematics uses this: the
  network already holds every joint's axis.

**In spikes.** One step of the recursion is: the spikes that carry an entry of $M$ and the
spikes that carry $\cos q_k$ (or $\sin q_k$) meet in a product population (A.6); the spikes
of two product populations meet in a sum population that holds an entry of $R_k$ (A.5).
110 of 200 product neurons fire (447 spikes per tick), 30 of 60 sum neurons (148).

**Built.** SO-3 `ring_so3/wiring.py` writes the recursion as a list of steps (`Product`,
`Entry`); the spiking network and an exact evaluation follow the same list;
`ring_so3/nest/network.py` builds it.

**Measured.** See C.5.

## C.3 From a bump to cos and sin

**What it is for.** The crossing from a place code (the joint's ring) to values (its cosine
and sine).

**Mathematics.** A pair population reads its joint's ring. Ring neuron $j$ stands for the
joint angle $q_j$ and projects with the weights $\eta\cos q_j$ and $\eta\sin q_j$ onto the two
coordinates of the pair. What arrives is

```math
\eta\sum_j r_j\begin{pmatrix}\cos q_j\\ \sin q_j\end{pmatrix} .
```

For a bump of even shape $b$ at the angle $q$, $r_j = b(q_j - q)$. With $u = q_j - q$ and the
addition theorems, the terms odd in $u$ cancel (as in B.4):

```math
\sum_j b(q_j - q)\begin{pmatrix}\cos q_j\\ \sin q_j\end{pmatrix} = B_1\begin{pmatrix}\cos q\\ \sin q\end{pmatrix}, \qquad B_1 = \sum_u b(u)\cos u .
```

The bump's width changes only the length $B_1$, not the direction. $\eta = 1/B_1$ is
measured once from the ring's own activity profile, so the pair has length 1. For a joint
whose range lies on 70% of the ring, $q_j$ is the joint angle of that index (B.1), continued
linearly beyond the arc.

*The readout normalises.* The pair population's decoders are fitted on a real ring whose
bump is written at 146 positions, with the target $(\cos q, \sin q)$ of the written angle.
A bump is a little stronger or weaker from tick to tick; fitted this way the readout gives
a pair of length 1 for the bump as it really is. This is how SO(3) deals with the amplitude
sensitivity of B.5.

**In spikes.** Every ring spike goes to all 120 neurons of the pair population with

```math
\mathsf w_{ij} = \frac{\alpha_i}{\varrho}\;\epsilon_i\cdot\begin{pmatrix}\cos q_j\\ \sin q_j\end{pmatrix}\frac{\eta}{e\,\tau_{15}}, \qquad \tau_{15} = 15\ \text{ms} ,
```

positive when the neuron's direction points toward the angle of the ring neuron that fired.
A ring spike is a vote of size $\eta = 0.001$ for its neuron's angle; the bump casts about
1,020 votes a second. The synapse is slower than elsewhere because a ring neuron fires only
20 to 60 times a second (A.2).

**Built.** SO-3 `nest/network.py`, `nest/calibrate.py`.

**Measured.** cos and sin within 0.018. 71 of 120 neurons fire, at 81 Hz, 287 spikes per
tick; weights of median size 7 pA, largest 135 pA.

## C.4 Lift, pitch, yaw and the output rings

**What it is for.** To show the orientation again as three place codes, and to define the
three angles a goal is given in.

**Mathematics.** The three angles are defined by

```math
R = \operatorname{Rot}(z,\psi)\operatorname{Rot}(y,\vartheta)\operatorname{Rot}(x,\phi)
= \begin{pmatrix} c_\psi c_\vartheta & \cdot & \cdot\\ s_\psi c_\vartheta & \cdot & \cdot\\ -s_\vartheta & c_\vartheta s_\phi & c_\vartheta c_\phi\end{pmatrix}
```

```math
\Longrightarrow\quad \vartheta = -\arcsin r_{31}, \qquad \psi = \operatorname{atan2}(r_{21}, r_{11}), \qquad \phi = \operatorname{atan2}(r_{32}, r_{33}) .
```

The pair $(r_{11}, r_{21}) = c_\vartheta(\cos\psi, \sin\psi)$ points in the direction of the
yaw and has the length $\cos\vartheta$; likewise $(r_{33}, r_{32})$ for the lift. At
$\vartheta = \pm90°$ both pairs have length 0: lift and yaw have no value there.

*A ring that shows the angle of a pair.* Each output ring has 100 neurons with preferred
angles $\theta_j = 2\pi j/100$. Neuron $j$ is driven by a pair $(x, y)$ of decoded values:

```math
J_j = I_{th} + g_o\big(x\cos\theta_j + y\sin\theta_j - h\big), \qquad h = 0.25 .
```

Write $(x, y) = \ell_p(\cos\omega, \sin\omega)$. Then
$x\cos\theta_j + y\sin\theta_j = \ell_p\cos(\theta_j - \omega)$: the drive is largest for the
neuron whose preferred angle is $\omega$. Neurons with $\ell_p\cos(\theta_j - \omega) > h$
fire: a bump centred on $\omega = \operatorname{atan2}(y, x)$ of half-width
$\arccos(h/\ell_p)$.

| Ring | Pair $(x, y)$ | Bump at |
| --- | --- | --- |
| yaw | $(r_{11}, r_{21})$ | $\psi$ |
| lift | $(r_{33}, r_{32})$ | $\phi$ |
| pitch | $(\cos\vartheta, -r_{31})$ | $\vartheta$ |

$\cos\vartheta$ is not an entry of the matrix. It is the length of the pair
$(r_{11}, r_{21})$: one more pair population holds that pair and its decoders are fitted to
the *length* of what it holds, $\sqrt{x^2 + y^2}$ (a function of the pair, A.5).

*Silence near the singular pose.* For lift and yaw $\ell_p = \cos\vartheta$. When
$\cos\vartheta < h$ no neuron is above threshold and the ring is silent: the network's way of
saying that the angle has no value.

**In spikes.** This is the crossing from values back to a place code, and the same idea as
the sheet's anchor (D.4): numbers in, a bump out, no place code on the way in. 34 of 100
neurons fire around the angle, the one in the middle at up to 148 Hz; 43 to 228 spikes per
tick, fewer when the pair is short. The angle is read as the population vector over four
ticks.

**Built.** SO-3 `kinematics.lift_pitch_yaw`, `nest/network.py`. $g_o$ is set so that the
peak neuron fires at 150 Hz for $\ell_p = 1$.

**Measured.** Where the angles are well defined (cos(pitch) > 0.3) no ring of any model is
off by more than 2.37° at the 90th percentile.

## C.5 The error, and how it grows along the arm

**What it is for.** One number for "how wrong is a decoded orientation", and a model of how
it accumulates.

**Mathematics.** For a decoded rotation $\hat R$ and the true $R$, the rotation that takes
one into the other is $\hat R R^\top$. A rotation by $\Delta$ about any axis has the trace
$1 + 2\cos\Delta$, so

```math
\Delta = \arccos\frac{\operatorname{tr}(\hat R R^\top) - 1}{2} .
```

*Nearest rotation.* The decoded matrix $M$ is close to a rotation, not exactly one. The
closest rotation in the least-squares sense comes from $M = U\Sigma V^\top$:
$\hat R = U\operatorname{diag}(1, 1, \det(UV^\top))V^\top$.

*Chance level.* For a rotation drawn uniformly from SO(3) the angle has the density
$(1 - \cos\Delta)/\pi$ on $[0, \pi]$, so

```math
\mathbb E[\Delta] = \frac1\pi\int_0^\pi\Delta(1 - \cos\Delta)\,d\Delta = \frac1\pi\Big(\frac{\pi^2}{2} + 2\Big) = \frac\pi2 + \frac2\pi \approx 126.5° ,
```

using $\int_0^\pi\Delta\cos\Delta\,d\Delta = [\Delta\sin\Delta + \cos\Delta]_0^\pi = -2$.

**Model: growth along the arm.** Let stage $k$ add an independent small error to the two
columns it writes. The stages are linear in the orientation so far, and later joints carry
an earlier error on as a rotation of the same size. If each stage adds an error of variance
$\varepsilon_s^2$ in a random direction, variances add:

```math
\Delta_n \approx \sqrt{\Delta_1^2 + (n-1)\,\varepsilon_s^2} .
```

**In spikes.** Nothing pulls the matrix back onto a rotation between stages, so its length
error accumulates too. A decoder is fitted with all neurons present; removing neurons
removes their terms from $\sum_i\delta_i r_i$, the value is off by an amount that does not
average to zero, and every later stage inherits it.

**Built.** SO-3 `ring_so3/kinematics.py`.

**Measured** (300 random held poses per model, SO-3 `docs/results.md`):

- end effector: 0.43° in the median for two joints, 1.06° (generic) and 1.14° (TIAGo) for
  seven; the largest 90th percentile of any model 1.80°;
- links of `generic7`: 0.21°, 0.47°, 0.60°, 0.74°, 0.85°, 0.98°, 1.06°; the formula with
  $\Delta_1 = 0.21°$, $\varepsilon_s = 0.42°$ gives 0.21°, 0.47°, 0.63°, 0.76°, 0.87°, 0.96°,
  1.05°;
- with 5% of the product neurons off: 5.25° in the median for seven joints.

**To derive further.** $\varepsilon_s$ from the product's error (A.6): how an error of 0.0077
in each of 12 products per stage becomes an angle.

## C.6 A moving arm: delay and writing ahead

**Model.** Every stage is a filter (A.7). Together the readout is the true orientation of a
time $\tau$ earlier, the same at every speed. If the hand turns at the angular speed
$\Omega$, the error while moving is $\Delta_{\text{moving}} \approx \Omega\,\tau$.

*Writing ahead.* A constant delay can be taken out at the input. To first order
$q_k(t + \tau_k) \approx q_k(t) + \tau_k\dot q_k(t)$, so the encoder of joint $k$ is given

```math
\tilde q_k = \operatorname{clip}(q_k + \tau_k\dot q_k), \qquad \tau_k = 0.14\ \text{s} + 0.01\ \text{s}\cdot(n - k) :
```

an early joint passes through more stages and gets a longer lead.

**Measured.** $\tau$ = 0.16 to 0.20 s (bump 0.09 s behind a moving angle; 15 ms synapse;
a few ms per population; 50 ms readout cells). At 0.5 rad/s per joint the seven-joint
readout is 6.85° off the orientation of the same instant and 1.46° off that of one delay
earlier. With writing ahead: delay 0.03 s, 1.80° while moving.

## C.7 Comparing two orientations

**What it is for.** The inverse task: a goal orientation is given as three angles on three
goal rings; wanted is how the hand still has to turn.

**Mathematics.** There is one rotation that takes $R$ to $G$:
$G R^\top = \operatorname{Rot}(\hat n, \Delta)$, about a unit axis $\hat n$ by an angle
$\Delta\in[0,\pi]$. With $r_i$, $g_i$ the columns of $R$ and $G$, define

```math
e = \tfrac12\sum_{i=1}^3 r_i\times g_i .
```

*Claim:* $e = \sin\Delta\,\hat n$.

*Derivation.* For vectors $a, b$ the matrix $ba^\top - ab^\top$ is antisymmetric, and it is
the cross-product matrix of $a\times b$: $[a\times b]_\times = ba^\top - ab^\top$ (check on
$a = \hat x$, $b = \hat y$: $\hat y\hat x^\top - \hat x\hat y^\top$ has the entries
$(2,1) = 1$, $(1,2) = -1$, which is $[\hat z]_\times$). Summing over the columns,

```math
\Big[\sum_i r_i\times g_i\Big]_\times = \sum_i\big(g_i r_i^\top - r_i g_i^\top\big) = G R^\top - R G^\top .
```

Rodrigues' formula is
$\operatorname{Rot}(\hat n,\Delta) = I + \sin\Delta\,[\hat n]_\times + (1 - \cos\Delta)[\hat n]_\times^2$;
$I$ and $[\hat n]_\times^2$ are symmetric and $[\hat n]_\times$ antisymmetric, so

```math
G R^\top - R G^\top = \operatorname{Rot}(\hat n,\Delta) - \operatorname{Rot}(\hat n,\Delta)^\top = 2\sin\Delta\,[\hat n]_\times .
```

Hence $\sum_i r_i\times g_i = 2\sin\Delta\,\hat n$. ∎

So $e$ points along the axis about which the hand has to turn; its length is about the
angle for a small turn, never more than 1, and zero at $\Delta = 180°$ (which the travelling
goal of C.11 avoids).

*The goal matrix.* $G = \operatorname{Rot}(z,\psi^\ast)\operatorname{Rot}(y,\vartheta^\ast)\operatorname{Rot}(x,\phi^\ast)$
is the orientation of a three-joint chain with the axes z, y, x. The forward network's own
wiring (C.2) builds it from the cos and sin of the three goal rings.

*A second comparison, through the angles* (built, not used for the pose): with
$\cos\vartheta$ from C.4,

```math
d_\phi = \sin\phi^\ast r_{33} - \cos\phi^\ast r_{32} = \cos\vartheta\sin(\phi^\ast - \phi), \quad
d_\vartheta = \sin\vartheta^\ast\cos\vartheta + \cos\vartheta^\ast r_{31} = \sin(\vartheta^\ast - \vartheta), \quad
d_\psi = \sin\psi^\ast r_{11} - \cos\psi^\ast r_{21} = \cos\vartheta\sin(\psi^\ast - \psi) ,
```

turned into a vector as $e = d_\phi r_1 + d_\vartheta(-r_{21}, r_{11}, 0) + d_\psi\hat z$. It
agrees with the first for small differences at level pitch and shrinks toward pitch ±90°.

**In spikes.** Each component of $e$ is a sum of six products of one entry of $R$ and one of
$G$: 18 product populations (437 spikes per tick each) and three sum populations that hold
$2e$ (132 spikes per tick while a turn is asked; at the goal 93: a turn of zero is a firing
pattern). The goal matrix costs 14 products and 4 sums. A turn larger than 0.5 rad per
component is held at 0.5 rad.

**Built.** SO-3 `ik/rules.py: turn_from_matrices`, `turn_from_angles`; `ik/wiring.py`.

**Measured.** Direction of the comparison within 5.0° in the median on the worst of 7 runs
(mark 10°).

## C.8 The cascade

**What it is for.** To turn the error into one number per joint without solving anything.

**Mathematics.** Joint $k$ turns link $k$ and everything after it about its axis $a_k$. If
the joints move by small $\delta q_k$ the hand turns by

```math
\delta\omega = \sum_{k=1}^n a_k\,\delta q_k .
```

The cascade hands the turn down the arm:

```math
e_1 = e, \qquad t_k = \operatorname{clip}(a_k\cdot e_k,\ \pm\bar t_k), \qquad e_{k+1} = e_k - a_k t_k .
```

$t_k$ is joint $k$'s *share*: the part of what is left that lies about its own axis.

*The shares add up to the turn.* Summing $e_{k+1} = e_k - a_k t_k$ over $k$:

```math
\sum_{k=1}^n a_k t_k = e_1 - e_{n+1} = e - e_{n+1} .
```

If the joints move in proportion to their shares, $\delta q_k = \kappa\,t_k\,\delta t$, the
hand turns by $\kappa\,\delta t\,(e - e_{n+1})$: exactly along the turn asked for, less what no
joint could take.

*What is left shrinks.* Without caps $e_{k+1} = (I - a_k a_k^\top)e_k$ is $e_k$ with its part
along $a_k$ removed, so $\lVert e_{k+1}\rVert\le\lVert e_k\rVert$, and after three joints with
perpendicular axes nothing is left.

*The path.* When nothing is left the hand's angular velocity is $\kappa\,e = \kappa\sin\Delta\,\hat n$,
about the fixed axis $\hat n$: $\dot\Delta = -\kappa\sin\Delta$, the shortest turn.

*Why not give every joint the whole error.* Then $t_k = a_k\cdot e$ and the hand turns by
$\kappa\,\delta t\,(\sum_k a_k a_k^\top)e$. The matrix $\sum_k a_k a_k^\top$ has the trace $n$:
with seven joints the turn is made up to seven times over along some directions and less
along others. The gain must be lowered for the direction served most, and the path bends
toward it.

*At a cap or a joint limit* the joint takes less and the rest stays in $e_{k+1}$. With
$\zeta\in[0,1]$ falling from 1 to 0 over the last 5 ring indices before a limit, the share
that points beyond it is reduced as

```math
|t_k| \leftarrow \max\big(|t_k| - (1 - \zeta)\,\bar t_k,\ 0\big) .
```

**In spikes.** See C.9.

**Built.** SO-3 `ik/rules.py: cascade, faded, limit_room`; the identity is tested.

**Measured** (exact rule, 100 pairs per arm, SO-3 `docs/ik/results/tables.md`): cascade
reaches 100% on `generic7` (5° after 3.7 s, path 1.23 times the shortest); all joints at
once 100% (3.8 s, 1.31); one joint at a time 36%.

**To derive further.** Convergence of the closed loop with caps; the path when something is
left over.

## C.9 The cascade as populations

**What it is for.** The rule of C.8 as products and sums.

**Mathematics.** In what the populations hold:

| Value | Held by | Fed with |
| --- | --- | --- |
| $2e$ | 3 sum populations | 18 products $r_{ai}g_{bi}$ |
| $a_k\cdot e_k$ | 3 products per joint (none for joint 1, whose axis is constant) | link entries, what is left |
| share | two **half** populations, $h^+_k = [x]_0^1$, $h^-_k = [-x]_0^1$, $x = a_k\cdot e_k/\bar t_k$ | the products |
| $2e_{k+1}$ | 3 sum populations | $2e_k - 2\bar t_k\,a_k(h^+_k - h^-_k)$: 3 more products |

The share is $t_k = \bar t_k(h^+_k - h^-_k)$. Two halves and not one signed value, because
each half drives one side of the joint's gain pair (C.10) and each is held down separately
near its joint limit.

**In spikes.**

- A half is a sum population of 60 neurons read with decoders fitted to $[x]_0^1$ (A.5).
  Both halves fire all the time, about equally (330 and 310 spikes per tick at a full
  positive share); what differs is which neurons fire and how the next stage reads them:
  from the + half 1, from the − half 0.
- The sums that carry $2e$ and what is left use a readout that stays at ±1 outside its
  range, so a large turn fills them and does not wrap.
- **The hold at a limit.** The joint's own ring also projects onto its two halves. Ring
  neurons within 5 indices of the upper limit inhibit the + half with a weight that rises
  toward the limit; those near the lower limit the − half. A bump in that zone lowers the
  value the half holds, and at the limit it reads 0 whatever the share:
  $|t_k|\leftarrow\max(|t_k| - (1-\zeta)\bar t_k, 0)$ is this subtraction.
- From the third joint on a joint costs 6 products, 3 sums and 2 halves: 1,500 neurons.

**Built.** SO-3 `ik/wiring.py`, `nest/ik_network.py`, `nest/ik_calibrate.py`.

**Measured.** Right sign in 100% of the shares above 0.05 rad (mark 95%); the error of a
share does not grow along the arm.

## C.10 A joint's goal ring

**What it is for.** The crossing from a value (the share) back to a place code: a goal ring
whose activity leads the joint's state ring by the share. The joint follows it.

**Mathematics.** Wanted: a *lead* $\ell$ (ring indices) that is zero without a share, has
the share's sign down to shares of 0.02 rad, and grows with it up to a cap.

*Built: a ring that is written.* One gain pair, gated by the *state* bump:

- gain neuron $i$ of both sides gets the spikes of state neuron $i$ and a constant current;
  under the state bump both sides fire at a resting rate, elsewhere they are silent;
- the share moves the two sides apart: the right side gets $+D(h^+ - h^-)$, the left side
  $-D(h^+ - h^-)$;
- right neuron $i$ excites goal neuron $i + m$, left neuron $i$ goal neuron $i - m$, $m = 8$.

The goal ring's activity is the state bump copied $m$ indices up with the weight $r_R$ and
$m$ indices down with the weight $r_L$. The centre of two copies of the same shape at $\pm m$
with the weights $r_R, r_L$ is

```math
\ell = \frac{(+m)\,r_R + (-m)\,r_L}{r_R + r_L} = m\,\frac{r_R - r_L}{r_R + r_L} .
```

Without a share $r_R = r_L$ and $\ell = 0$ by symmetry, whatever the rates are. A full share
silences one side: $\ell = \pm m$, the cap. So the largest share is the largest lead:
$\bar t_k = m\,\rho_{q,k}$. Both sides fire at rest, so a small share has no threshold to
cross.

**In spikes.** What a gain neuron (rheobase 759,375 pA) receives:

| Input | Size | In units of the rheobase |
| --- | ---: | ---: |
| spikes of state neuron $i$ (weight 137,000 pA) at 68 Hz | 683,735 pA | 0.90 |
| a constant current | 304,000 pA | 0.40 |
| the two halves, through decoder weights, $\pm D(h^+ - h^-)$, $D$ = 228,000 pA | −228,000 to +228,000 pA | −0.30 to +0.30 |

- Neither the ring spikes (0.90) nor the constant (0.40) alone fires it; under the centre of
  the state bump it gets both: 1.30 at rest (97 Hz by A.1), 1.60 and 1.00 at a full share.
- A goal-ring neuron has no bias and no recurrent weights; it fires only while gain spikes
  arrive (weight 84,000 pA). At rest the goal ring shows two lumps of 11 neurons, 8 above
  and 8 below the state bump; with a full share one lump at about 200 Hz.
- The lead is read as the population vector of the goal ring's spikes over three ticks.

**Built.** SO-3 `nest/goal_ring.py: build_written_ring`. An attractor goal ring with a
share pair and a body pair (the picture of B.10) was tried first and met no setting's marks:
a bump of an attractor moves only above a threshold and then by more than an index, so a
graded lead below one index does not come out of it.

**Measured.** Lead for the shares 0, +0.3, +1, 0, −0.6 of the cap: −0.1, +3.2, +8.0, +0.1,
−7.1 indices. Gain spikes per tick per side: 54 at rest, 84 and 13 at a full share. After a
step of the share the lead has covered 63% of its change after 0.20 to 0.30 s. The lead is
about 1.4 times steeper than the share in the middle of its range.

**To derive further.** $\ell$ as a function of the share from $f(J)$ of the two sides (the
curve $r_R(x), r_L(x)$ and its slope at zero), and the 0.2 to 0.3 s from the 27 and 50 ms
synapses and the three-tick readout.

## C.11 Closing the loop

**What it is for.** To say when the loop is stable, how fast it can be, and where it stops.

**Mathematics.** The light robot moves every joint toward the angle its goal ring holds,
$\dot q_k = \kappa(\hat q^G_k - q_k)$; with $\hat q^G_k - q_k\approx t_k$ this is
$\dot q_k = \kappa t_k$. From C.8, while nothing is left over and the turn is small, and with
the comparison and the lead late by $\tau$ in all,

```math
\dot\Delta(t) = -\kappa\,\Delta(t - \tau) .
```

*Stability.* Try $\Delta = e^{st}$: $s = -\kappa e^{-s\tau}$. On the stability boundary
$s = \mathrm i\omega$: $\mathrm i\omega = -\kappa(\cos\omega\tau - \mathrm i\sin\omega\tau)$,
so $\cos\omega\tau = 0$ and $\omega = \kappa\sin\omega\tau$, that is $\omega\tau = \pi/2$ and
$\omega = \kappa$:

```math
\text{stable for}\quad \kappa\tau < \frac\pi2 .
```

*No overshoot.* Real solutions $s = -\sigma$ need $\sigma = \kappa e^{\sigma\tau}$, i.e.
$\sigma\tau e^{-\sigma\tau} = \kappa\tau$. The left side is at most $1/e$ (at
$\sigma\tau = 1$), so a real, non-oscillating solution exists only for

```math
\kappa\tau \le \frac1e \approx 0.37 .
```

- **Rate limit.** A joint takes at most $\bar t_k$ = 8 indices (0.25 rad on a ring that
  covers the circle), so it moves at most at $\kappa\bar t_k$.
- **Dead zone.** If shares below $d_0$ move nothing, the loop stops while every
  $|a_k\cdot e_k| < d_0$; the hand rests up to about $d_0$ per independent direction from the
  goal.
- **A goal that travels.** A bump does not jump (B.2), and $e$ is zero for a turn of 180°.
  So the three goal angles move toward a new goal at up to 1 rad/s, lift and yaw the short
  way round; the comparison sees the goal a little ahead of the hand.

**In spikes.** The decoded shares never fall silent, so a trial ends when the hand has
turned less than 0.5° within 2 s. No spike goes to the robot: the goal ring's angle is read
from spike counts once per tick.

**Built.** SO-3 `ik/session.py`, `ik/rules.py: travel`; $\kappa$ = 1.25 per second.

**Measured** (spiking network, light robot, SO-3 `docs/ik/results/tables.md`): `generic7`
100% of 67 goals reached, final error 1.32° (median), 5° within 4.5 s, path 1.20 times the
shortest; `tiago7` 94% of 83, 1.32°, 5.5 s, 1.19.

**To derive further.** The delay equation with the rate limit (a saturating loop) and with
the lead's own first-order lag in place of a pure delay.

## C.12 The robots

Both answer one call, `follow(targets, time_s, dt)`: "move toward these joint angles as if
to arrive after `time_s`, for one tick".

- **Light robot.** $q\leftarrow q + (q^{\text{target}} - q)\,\delta t/T$ inside the joint
  limits; with the target the goal ring's angle and $T = 1/\kappa$ this is
  $\dot q = \kappa(\hat q^G - q)$.
- **TIAGo in Gazebo.** The targets go to the arm's trajectory controller as one point to be
  reached after $T$; the controller interpolates from its own set-point to the target (the
  same proportional approach) and a PID per joint tracks it against gravity. Then exactly
  one tick of physics passes: physics is paused, and every step unpauses it, waits until
  Gazebo's clock has advanced by 50 ms, and pauses again. So the network may be as slow as
  it likes.

**Built.** SO-3 `ring_so3/robots.py`, `ring_so3/gazebo/link.py`.

## C.13 The standard methods, and where the cascade sits

With $J = (a_1\ \cdots\ a_n)$, so that $\delta\omega = J\,\delta q$:

- **Jacobian transpose**, $t = J^\top e$. Steepest descent of $\tfrac12\lVert e\rVert^2$: a
  step $t$ changes $e$ by $-Jt$, so $\tfrac12\lVert e\rVert^2$ changes by
  $-e^\top JJ^\top e = -\lVert J^\top e\rVert^2\le0$. It never divides. The hand turns by
  $JJ^\top e$, not by $e$.
- **Damped least squares**, $t = \arg\min_t\lVert Jt - e\rVert^2 + \lambda^2\lVert t\rVert^2$.
  Setting the gradient to zero, $(J^\top J + \lambda^2I)t = J^\top e$; since
  $J^\top(JJ^\top + \lambda^2I) = (J^\top J + \lambda^2I)J^\top$,

  ```math
  t = J^\top(JJ^\top + \lambda^2I)^{-1}e = \sum_i\frac{\sigma_i}{\sigma_i^2 + \lambda^2}\,v_i\,(u_i\cdot e)
  ```

  with the singular values and directions of $J$. For $\lambda = 0$ the factor is
  $1/\sigma_i$ (the pseudo-inverse, unbounded as $\sigma_i\to0$); with $\lambda > 0$ it is at
  most $1/(2\lambda)$.
- **Cyclic coordinate descent.** Joint after joint, the angle that brings the hand closest
  with the others held: $x^\ast = \operatorname{atan2}(s, c)$ with
  $c = \operatorname{tr}M - a^\top Ma$, $s = \operatorname{tr}([a]_\times M)$, $M = RG^\top$;
  for a small turn $x^\ast\approx a\cdot e$.

**The cascade is one Gauss-Seidel sweep on** $\min_t\lVert Jt - e\rVert^2$. Minimising over
$t_k$ alone gives $t_k = a_k\cdot(e - \sum_{j\ne k}a_jt_j)/\lVert a_k\rVert^2$. Starting from
$t = 0$ and going through the joints once, only the earlier $t_j$ are non-zero when joint
$k$ is reached, and $\lVert a_k\rVert = 1$: exactly $t_k = a_k\cdot e_k$. In the loop a new
sweep is made every tick from the new pose. Against the transpose it differs by the
subtraction alone; against damped least squares it makes the same turn with more joint
motion and needs no inverse.

**Built.** SO-3 `ring_so3/standard/ik.py`.

---

# Part D. SE(3): position on sheets, and the pose

Repo: `Ring-Attractor-SE-3-Joint-Architecture` (package `ring_se3`, with a copy of
`ring_so3`). Its reference: this repo's `TECHNICAL.md`, parts I to III.

## D.1 The position is a weighted sum

**What it is for.** The pose is a rotation and a translation,
$T = \begin{pmatrix}R & p\\ 0 & 1\end{pmatrix}$. Part C gives $R_k$ for every link. This adds
$p$ without a single product.

**Mathematics.** Joint $k$ sits in link $k-1$ at the fixed offset $o_k$ (a vector in that
link's frame); the end point sits in link $n$ at $o_{n+1}$. The origin $c^{o}_k$ of joint $k$
is reached from that of joint $k-1$ by going $o_k$, turned by that link's orientation:

```math
c^o_k = c^o_{k-1} + R_{k-1}\,o_k, \quad c^o_0 = 0
\qquad\Longrightarrow\qquad
c^o_k = \sum_{m=0}^{k-1}R_m\,o_{m+1}, \qquad p = \sum_{m=0}^n R_m\,o_{m+1} .
```

By components $p_i = \sum_m\sum_j(R_m)_{ij}(o_{m+1})_j$: a sum of matrix entries of the
links, each times a **constant**. The first term, $R_0o_1 = o_1$, is the origin of the first
joint, which does not move.

*The lever of joint $k$* is the vector from its origin to the end point:

```math
l_k = p - c^o_k = \sum_{m=k}^n R_m\,o_{m+1}, \qquad l_1 = p - o_1 .
```

*Scaling.* A population holds numbers in $[-1,1]$. A rotation preserves length, so
$\lVert R_mo_{m+1}\rVert = \lVert o_{m+1}\rVert$, and by the triangle inequality

```math
\lVert p - o_1\rVert\le L = \sum_{m\ge1}\lVert o_{m+1}\rVert, \qquad \lVert l_k\rVert\le L_k = \sum_{m\ge k}\lVert o_{m+1}\rVert .
```

The network holds $u = (p - o_1)/L$ and $l_k/L_k$. $L$ is the *reach*, the radius of the
workspace; position errors are given as a share of it.

**In spikes.** Three sum populations, the *position cells*, one per coordinate. The spikes
of the populations that hold the link entries arrive with the weights of A.5 times
$(o_{m+1})_j/L$. No cell is added to the orientation network for this, only synapses. A
lever is three more sum populations per joint.

**Built.** `ring_se3/kinematics.py` (exact), `ring_se3/wiring.py` (the list of sums),
`ring_se3/chains.py` (reach, workspace).

**Measured.** $L$ = 0.806 m for the 7-joint TIAGo arm. Position cells alone: 1.28% of the
reach in the median (7 joints).

## D.2 The velocity of the end point

**What it is for.** The path-integration input (D.5), and the same quantity that the pose
IK needs (D.7).

**Mathematics.** Hold all joints but joint $k$ still. Everything beyond it is one rigid
body turning about the axis $a_k$ through the point $c^o_k$ at the angular speed $\dot q_k$.
A point $x$ of a rigid body turning with angular velocity $\Omega$ about a point $c$ moves
with $\Omega\times(x - c)$, so

```math
\frac{\partial p}{\partial q_k}\,\dot q_k = \dot q_k\,a_k\times(p - c^o_k) = \dot q_k\;a_k\times l_k ,
\qquad
v = \dot p = \sum_{k=1}^n\dot q_k\,(a_k\times l_k) .
```

$a_k\times l_k$ is column $k$ of the position Jacobian. Scaled, with
$\hat v = v/(L\,\dot q_{\text{ref}}\,\nu)$, $\dot q_{\text{ref}}$ = 1 rad/s, $\nu$ = 1.5:

```math
\hat v = \sum_k\frac{L_k}{L\,\nu}\Big(\tfrac{\dot q_k}{\dot q_{\text{ref}}}a_k\Big)\times\frac{l_k}{L_k} .
```

**In spikes.** Nine products per joint: three for $\dot q_k a_k$, six for the cross product
(two per component). Joint 1 turns about a fixed axis, so its first three cost nothing: 8,
17, 26, 35, 44, 53 products for 2 to 7 joints.

**Built.** `ring_se3/wiring.py`, `ring_se3/nest/network.py`. The network with the IK layer
is built without this path.

**Measured.** The velocity populations are within 4% to 5% of the true speed for a resting
arm.

## D.3 The attractor sheet

**What it is for.** To hold a position as the place of a packet: the two-dimensional
version of the ring, after RatSLAM's pose cells.

**Mathematics.** A sheet is a square of $n_s\times n_s$ neurons ($n_s$ = 48) over one
coordinate plane. Neuron $(i, j)$ stands for the place

```math
c_{ij} = \big(-E_s + (i + \tfrac12)\ell_c,\; -E_s + (j + \tfrac12)\ell_c\big), \qquad \ell_c = \frac{2E_s}{n_s} = 0.05, \quad E_s = 1.2 ,
```

in scaled coordinates: one cell is 5% of the reach, and $E_s > 1$ leaves room at the edge.
Three sheets hold the planes $(x,y)$, $(x,z)$, $(y,z)$: every coordinate is on two sheets.

*Recurrent weights.* The ring's profile $w$ (B.2) over the distance in the plane:

```math
W_{a\to b} = w\big(\lambda_s\,\lVert c_b - c_a - \chi\,\ell_c\,\hat e_a\rVert/\ell_c\big), \qquad \lambda_s = 1.5, \quad \chi = 3 ,
```

with $\lambda_s$ the profile's distance units per cell and $\hat e_a$ the unit vector of
neuron $a$'s *label*. For $\chi = 0$ this is the ring's connectivity turned into a disc.

**Model.** As on the ring: neurons inside a group keep each other firing and silence their
surround; $W$ depends only on the difference of places, so the packet can sit anywhere: a
two-dimensional continuous attractor.

*Labels.* Every neuron carries one of four labels, $+a, -a, +b, -b$, in a 2 x 2 tiling. The
centre of its *outgoing* weights is moved $\chi$ cells along its label. Over a packet the
four groups are equally many and the shifts cancel,
$(1,0) + (-1,0) + (0,1) + (0,-1) = 0$, so the packet rests. D.5 breaks that balance on
purpose.

**In spikes.** The ring neuron (A.1) as `iaf_psc_alpha_multisynapse`, with four synapse
types: the ring's excitatory (27 ms) and inhibitory (50 ms), a fast inhibitory one for the
velocity cells (3 ms), and one for the anchor that carries both signs (20 ms). A packet is
about six cells across; about 42 of 2,304 neurons fire at 12 Hz in the mean, about 25 spikes
per tick.

**Built.** `ring_se3/sheet.py`, `ring_se3/nest/sheet.py`.

**Measured.** A packet left alone for 10 s: alive in 8 of 8; it moves 0.33 / 0.93 / 1.07
cells (median / 90th percentile / largest).

**To derive further.** As for the ring (B.2), in two dimensions; the lattice pinning (a
packet prefers certain places relative to the cells).

## D.4 The anchor

**What it is for.** The crossing from numbers to a place code: to pull the packet to a
given place $x$ (two scaled coordinates), using only sums of values that populations hold.

**Mathematics.** The neuron at the place $c$ receives a bowl turned upside down with its
top $b_a$ at $x$:

```math
I_c(x) = b_a - \frac{g_a}{2}\lVert c - x\rVert^2 .
```

Expanding the square, $\lVert c - x\rVert^2 = \lVert c\rVert^2 - 2c\cdot x + \lVert x\rVert^2$:

```math
I_c(x) = \underbrace{g_a\big(c\cdot x - \tfrac12\lVert x\rVert^2\big)}_{\text{moving part}} + \underbrace{\big(b_a - \tfrac{g_a}2\lVert c\rVert^2\big)}_{\text{fixed part}} .
```

- *Moving part.* $c\cdot x - \tfrac12\lVert x\rVert^2 = \sum_{a\in\{1,2\}}(c_ax_a - \tfrac12x_a^2)$:
  per coordinate, a weighted sum of $x_a$ and of $x_a^2$. Both are read from one *gate*
  population that holds $x_a$, with two sets of decoders $\delta^{(1)}$ (the value) and
  $\delta^{(2)}$ (its square): a function of the value, A.5.
- *Fixed part.* The same for every pose: a bias per neuron.

*Why the term $-\tfrac12\lVert x\rVert^2$ must be there.* Without it the current is
$b_a - \tfrac{g_a}2\lVert c - x\rVert^2 + \tfrac{g_a}2\lVert x\rVert^2$: still peaked at $x$
but lifted by $\tfrac{g_a}2\lVert x\rVert^2$ everywhere. These neurons rest at threshold, so
a positive current makes them fire: a whole disc around $x$ would be driven. With the term
the current is at most $b_a$ and positive only within $\sqrt{2b_a/g_a}$ of $x$.

**Model: the packet follows like a ball in a bowl.** A packet at $y$ sees a current that
falls across it with the slope $\nabla I = g_a(x - y)$. The neurons toward $x$ are driven
more, those on the other side less; the packet gains neurons on one side and loses them on
the other at a speed that grows with the slope:

```math
\dot y = \mu_a\,g_a\,(x - y) \quad\Longrightarrow\quad \tau_a = \frac{1}{\mu_ag_a} .
```

**An error of the square moves the whole bowl.** If the decoded $x^2$ is off by $\epsilon_2$,
the current changes by $-\tfrac{g_a}2\epsilon_2$ at *every* neuron: the place of the peak is
untouched, the height is not. This is what level cells correct (D.12).

**In spikes.**

- The weight from gate neuron $j$ to the sheet neuron at $c$ is
  $g_a\,(c_a\delta^{(1)}_j - \tfrac12\delta^{(2)}_j)/(e\,\tau_{\text{anchor}})$: positive or
  negative depending on where the receiver sits.
- The fixed part is delivered by 16 *bias cells* that fire at a steady 99 Hz, not as a bias
  current. Two findings, both by measurement:
  - *both parts through one synapse type.* Through the neuron's two usual synapses the
    negative part would decay with 50 ms and the positive with 27 ms: whenever the anchor
    changes, a net negative current is left over and silences the packet;
  - *both parts through spikes.* The two parts are each thousands of pA far from the
    centre and cancel to a small sum. If one were switched instantly and the other faded
    with its synapse, the whole sheet would fire, then fall silent.
- Switching the anchor off silences the gates and the bias cells; the packet then lives on
  the sheet's own spikes.

**Built.** `ring_se3/nest/sheet.py: connect_anchor, connect_anchor_bias`. $g_a$ = 8 pA per
cell² (3,200 pA per scaled unit²), $b_a$ = 32 pA, so the current is positive within
$\sqrt{2\cdot32/8}\approx2.8$ cells of $x$.

**Measured.** A sheet anchored through gate populations at 51 places: off by 0.14 / 0.28 /
0.56 cells. With the anchor moving at 0.5 units/s the packet is 15 ms behind for $g_a$ = 8
and 40 ms for $g_a$ = 4, about inverse to $g_a$ as the model says. The square is read to
0.015 (root mean square), so the height varies by some tens of pA.

**To derive further.** $\mu_a$ from the sheet dynamics; the readout error of $x^2$ and the
resulting spread of the height.

## D.5 Path integration

**What it is for.** With the anchor off, to move the packet with the end point's velocity.
Built and measured; it works only roughly, and the IK does not use it.

**Mathematics.** For each coordinate and sign there are 8 *velocity cells*. A spike of a
cell for the direction $+a$ sends a short, strong inhibition to the sheet neurons labelled
$-a$ in the two sheets that hold $a$.

*Why silencing one group moves the packet.* While the group $-a$ is silent the remaining
shifts no longer cancel: $\sum\hat e = (+1, 0)$. The packet's outgoing excitation is centred
ahead of it and it moves that way. Silenced a fraction $D$ of the time, the mean speed is
about $D$ times the top speed.

*A cell whose rate is proportional to its input.* A velocity cell has a very long membrane
time constant ($\tau_m$ = 5 s). From A.1, for $J\gg I_{th}$,

```math
T = \tau_m\ln\frac{J}{J - I_{th}}\approx\tau_m\frac{I_{th}}{J} = \frac{C_mV_{th}}{J} \quad\Longrightarrow\quad f\approx\frac{J}{C_mV_{th}} :
```

it integrates its input and fires once per $C_mV_{th}$ = 3,750 pA·ms of charge.

*Push-pull about a resting rate.* Let $\Sigma(r)$ be the packet's speed when one group is
pulsed at the rate $r$. $\Sigma$ has a dead zone. So all velocity cells get a resting drive
$r_0$ while the arm moves, and a velocity raises one side by $\Delta_r$ and lowers the other:

```math
\text{speed} = \Sigma(r_0 + \Delta_r) - \Sigma(r_0 - \Delta_r)\approx2\,\Sigma'(r_0)\,\Delta_r .
```

The velocity population's decoders are fitted to the inverse of the measured speed curve.

**Model: why it is crude.** A packet has about 30 neurons that fire about 15 times a second;
one group is a quarter of them. A pulse silences that group for a few milliseconds; whether
that removes a spike depends on whether one of seven or eight neurons was about to fire. The
push per pulse is zero or one missing spike, by chance.

**Built.** `ring_se3/nest/calibrate.py`, `ring_se3/nest/sheet.py: connect_velocity_cells`.

**Measured.** A free packet driven by a known velocity moves at 0.66 to 1.71 times the speed
asked for (10th to 90th percentile; the mark "within 10%" is missed). The speed curve ends
at 9.47 cells per second. With the anchor off and the arm at rest a packet has moved 4.59%
of the reach in the median after 10 s.

## D.6 Reading the position; delay

**What it is for.** The crossing from a place code back to numbers, for tests and pages.
(Inside the network this is done by D.11.)

**Mathematics.** From the spike counts $n_c$ of one sheet,

```math
\hat x = \frac{\sum_cn_c\,c}{\sum_cn_c} ,
```

the centre of the spikes. Each coordinate is on two sheets; with independent errors of the
same variance $s^2$ the mean of two readings has $s^2/2$ and their difference $2s^2$:

```math
\hat p_x = \tfrac12\big(\hat x^{(xy)} + \hat x^{(xz)}\big), \qquad \text{disagreement}_x = \hat x^{(xy)} - \hat x^{(xz)} .
```

*Delay.* Let the packet obey $\dot y(t) = \hat v(t) + \tfrac1{\tau_a}\big(p(t-\tau) - y(t)\big)$:
pushed by the velocity and pulled toward an anchor that is the true position of a time
$\tau$ earlier. For $p(t) = vt$ and a perfect $\hat v = v$, try $y = vt - \epsilon_y$:

```math
v = v + \frac1{\tau_a}\big(v(t-\tau) - vt + \epsilon_y\big) \;\Longrightarrow\; \epsilon_y = v\,\tau .
```

The packet is behind by $v\tau$ whatever $\tau_a$ is and even with a perfect velocity input.
Anchoring ahead, $x_{\text{anchor}}(t) = p(t-\tau) + \tau\,\hat v(t-\tau)\approx p(t)$,
removes it.

**Measured** (300 held poses per model, 12 models, `docs/results/tables.md`):

- position from the sheets off by 0.97% to 1.37% of the reach in the median; the largest
  90th percentile 2.34%; the 7-joint TIAGo arm 11.1 / 18.6 / 32.8 mm;
- a single sheet is off by 0.71% to 1.22% in its plane, the two copies of a coordinate
  differ by only 0.30% to 0.34%: the errors are *not* independent; most of the error is that
  of the position cells that anchor all sheets, and averaging cannot remove it;
- a moving arm is followed 0.21 to 0.23 s behind at 0.5 rad/s; with the anchor 0.2 s ahead
  the delay is 0.03 s and the error while moving 2.28% instead of 5.26% (7 joints).

## D.7 The pose error, and what a joint does to it

**What it is for.** The inverse task for a pose: the goal is an orientation on three goal
rings and a position on three goal sheets.

**Mathematics.** Two comparisons:

```math
e = \tfrac12\sum_ir_i\times g_i = \sin\Delta\,\hat n \ \ (\text{C.7}), \qquad d = \frac{p^\ast - p}{L} = u^\ast - u .
```

A turn is in radians and an offset in reaches. One number, the *balance* $w$, says how many
radians of turn one reach of offset counts as:

```math
E = \begin{pmatrix}e\\ w\,d\end{pmatrix}\in\mathbb R^6, \qquad \lVert E\rVert^2 = \lVert e\rVert^2 + w^2\lVert d\rVert^2 .
```

*What one joint does.* By D.2, if joint $k$ moves by $\delta q_k$ the hand turns by
$a_k\delta q_k$ and its end point goes $(a_k\times l_k)\delta q_k$. In reaches per radian,

```math
m_k = \frac{a_k\times l_k}{L} = \rho_k\,c_k, \qquad c_k = a_k\times\frac{l_k}{L_k}, \qquad \rho_k = \frac{L_k}{L} .
```

$l_k/L_k$ lies in the unit ball, so $\lVert c_k\rVert\le1$ and $\lVert m_k\rVert\le\rho_k$.
Stacked,

```math
J_k = \begin{pmatrix}a_k\\ w\,m_k\end{pmatrix}, \qquad \lVert J_k\rVert^2 = 1 + w^2\lVert m_k\rVert^2 .
```

For all joints together the hand turns by $\sum_ka_k\delta q_k$ and its end point goes
$L\sum_km_k\delta q_k$.

**In spikes.** $a_k$ is a column of $R_k$ and $l_k/L_k$ a sum (D.1): both exist. New is the
cross product, two products per component, six per joint (joint 1: none, its axis is a
constant, so $c_1$ is a plain sum of position cells). A joint whose lever lies along its own
axis moves nothing (the last joint of the 7-joint TIAGo arm) and gets no $c_k$.

**Built.** `ring_se3/ik/rules.py: moves, lever_ratios, offset`.

**Measured.** $\rho_k$ for the 7-joint TIAGo arm: 1.00, 0.84, 0.73, 0.45, 0.24, 0.06, 0.06.

## D.8 The cascade over turn and offset

**What it is for.** One share per joint that serves turn and offset at once.

**Mathematics.** The rule:

```math
e_1 = e,\ d_1 = d, \qquad t_k = \operatorname{clip}\Big(\frac{a_k\cdot e_k + w^2\,m_k\cdot d_k}{N_k},\ \pm\bar t_k\Big), \qquad e_{k+1} = e_k - a_kt_k,\quad d_{k+1} = d_k - m_kt_k ,
```

in stacked form $t_k = J_k\cdot E_k/N_k$, $E_{k+1} = E_k - J_kt_k$.

*The share that leaves the least.* What is left after joint $k$ takes $t$ has the squared
length

```math
\lVert E_k - J_kt\rVert^2 = \lVert E_k\rVert^2 - 2t\,J_k\cdot E_k + t^2\lVert J_k\rVert^2 ,
```

a parabola in $t$. Its derivative $-2J_k\cdot E_k + 2t\lVert J_k\rVert^2$ vanishes at

```math
t^\star_k = \frac{J_k\cdot E_k}{\lVert J_k\rVert^2} = \frac{a_k\cdot e_k + w^2m_k\cdot d_k}{1 + w^2\lVert m_k\rVert^2}, \qquad \lVert E_{k+1}\rVert^2 = \lVert E_k\rVert^2 - \frac{(J_k\cdot E_k)^2}{\lVert J_k\rVert^2} .
```

So the best divisor is $N_k = \lVert J_k\rVert^2$, and what is left is perpendicular to
$J_k$. For orientation alone $m_k$ drops out, $N_k = 1$: the cascade of C.8.

*The shares add up.* Summing $E_{k+1} = E_k - J_kt_k$:

```math
\sum_ka_kt_k = e - e_{n+1}, \qquad \sum_km_kt_k = d - d_{n+1} ,
```

for any $N_k$ and any caps. If the joints move as $\delta q_k = \kappa t_k\delta t$, the hand
turns by $\kappa\delta t(e - e_{n+1})$ and its end point goes $\kappa\delta tL(d - d_{n+1})$.
Where nothing is left: $\dot p = \kappa(p^\ast - p)$, a straight line to the goal with the
time constant $1/\kappa$, and $\dot\Delta = -\kappa\sin\Delta$.

*What is left never grows* (exact divisor, no caps): every step removes
$(J_k\cdot E_k)^2/\lVert J_k\rVert^2\ge0$. Nothing is removed only if $J_k\cdot E = 0$ for
every joint, i.e. $J^\top E = 0$. Otherwise, with $\dot E = -\kappa(E - E_{n+1})$ for a small
turn and Cauchy-Schwarz,

```math
\frac{d}{dt}\tfrac12\lVert E\rVert^2 = -\kappa\,E\cdot(E - E_{n+1})\le-\kappa\lVert E\rVert\big(\lVert E\rVert - \lVert E_{n+1}\rVert\big)\le0 .
```

*The task "position".* The turn is left out: $t_k = w^2m_k\cdot d_k/N_k$ with the same
$N_k = 1 + w^2\lVert m_k\rVert^2$. The 1 now stands for the motion of the joint itself: this
is the share that minimises $w^2\lVert d_k - m_kt\rVert^2 + t^2$. Without it the best share
$m_k\cdot d_k/\lVert m_k\rVert^2$ grows without bound as the lever gets short.

**In spikes.** See D.13.

**Built.** `ring_se3/ik/rules.py: pose_cascade`; the identities are tested.

**Measured** (exact rule, 60 pairs per arm, `docs/ik/results/tables.md` section 1): cascade
95% to 100% of the goals, within the marks after 4.1 to 5.5 s; all joints at once 78% to
98%, 7.8 to 11.7 s; from the hand to the base 97% to 100%; two sweeps per tick 3.0 to 4.3 s.

**To derive further.** Convergence of the loop with one sweep per tick (it is one
Gauss-Seidel sweep on $J^\top Jt = J^\top E$ from $t = 0$, as in C.13); the size of
$E_{n+1}$ after one sweep for a given arm.

## D.9 Dividing by a constant

**What it is for.** $\lVert J_k\rVert^2 = 1 + w^2\lVert m_k\rVert^2$ changes with the pose,
between 1 and $1 + w^2\rho_k^2$, and the network cannot divide by a value.

**Mathematics.** Let $N_k$ be a constant of the arm. Then

```math
t_k = \frac{J_k\cdot E_k}{N_k} = \omega_kt^\star_k, \qquad \omega_k = \frac{\lVert J_k\rVert^2}{N_k} .
```

Putting $t = \omega_kt^\star_k$ into the parabola of D.8:

```math
\lVert E_{k+1}\rVert^2 = \lVert E_k\rVert^2 - 2\omega_kt^\star_kJ_k\cdot E_k + \omega_k^2t^{\star2}_k\lVert J_k\rVert^2 = \lVert E_k\rVert^2 - \omega_k(2 - \omega_k)\frac{(J_k\cdot E_k)^2}{\lVert J_k\rVert^2} .
```

- $\omega_k = 1$ removes the most;
- for $0 < \omega_k < 2$ something is removed;
- for $\omega_k > 2$ more is left than before, and the joints after it take it back, since
  the sums of D.8 hold whatever $N_k$ is.

This is the relaxation factor of successive over-relaxation.

*The mean is built.* $N_k = 1 + w^2\langle\lVert m_k\rVert^2\rangle$, with the mean over
4,000 poses drawn evenly from the joint ranges: a number that follows from the arm's
geometry, like the reach. Then $N_k = \langle\lVert J_k\rVert^2\rangle$ and $\omega_k$ is 1 in
the mean by construction.

**In spikes.** The division by $N_k$ is a factor on a synapse (D.13). Nothing else.

**Built.** `ring_se3/ik/rules.py: levers, normalisers`.

**Measured** (exact rule; `docs/ik/results/tables.md` sections 1 and 8):

| $N_k$ | Goals reached | Within the marks after |
| --- | --- | --- |
| the largest value, $1 + w^2\rho_k^2$ ($\omega_k\le1$) | 78% to 93% | 5.6 to 8.3 s |
| **the mean** | 95% to 100% | 4.1 to 5.5 s |
| the exact value (a division) | 95% to 100% | 4.0 to 5.0 s |

$N_k$ for the 7-joint TIAGo arm: 5.05, 5.03, 1.97, 2.66, 1.01, 1.03, 1.00; there $\omega_k$
lies between 0.20 and 1.93. For the 7-joint generic arm it reaches 3.99 and is above 2 in
5.2%, 6.0%, 1.1% and 5.4% of the poses for joints 1 to 4.

**To derive further.** A bound on $\lVert E_{n+1}\rVert/\lVert E\rVert$ for a sweep with
factors $\omega_k$, and a condition on the $\omega_k$ under which the closed loop still
contracts.

## D.10 The balance

$w$ enters twice: as the weight of the offset in the share, and in the divisor.

- **Which joint serves what.** A joint with $\lVert m_k\rVert = 1/w$ serves turn and offset
  equally. With $w = 3$ the base joints (levers near a reach) are moved almost only by the
  offset, the wrist joints almost only by the turn; the order of the cascade does the rest.
- **Where the hand ends when both cannot be met**: where
  $\lVert e\rVert^2 + w^2\lVert d\rVert^2$ is smallest. With $w = 3$ one degree of turn
  (0.01745 rad) is traded against $0.01745/3$ = 0.58% of the reach.
- **Against the marks.** Reached means within 5° (0.087 rad) and 4% of the reach;
  $w\cdot0.04 = 0.12$ rad: with $w = 3$ the two weigh about the same.

**Measured** (exact rule): with $w = 2$ the hand is within the marks later on every arm (5.7
to 7.6 s against 4.1 to 5.5 s); with 4 and 5 the 7-joint TIAGo arm reaches 93% and 90% of
the goals against 95%.

## D.11 From a packet to a number: the ramp readout

**What it is for.** The crossing from a place code on a sheet to values, *inside* the
network: the bridge from the two-dimensional packet to three numbers.

**Mathematics.** Give every sheet neuron a weight that rises along one coordinate, its own
place $c_{j,a}$:

```math
\mathcal R_a = \sum_jr_j\,c_{j,a}, \qquad \mathcal S = \sum_jr_j .
```

$\mathcal R_a$ is the ramp of B.6, now with the neuron's place as its weight. A sheet over
the plane $(a, b)$ gives two ramps from the same spikes with two sets of weights; that is
all the second dimension costs.

*Why the ramp alone is not enough.* For a packet at $x$ with the strength $\mathcal S$,
$\mathcal R_a = \mathcal Sx_a$: place **and** strength (B.5). The place itself is the ratio

```math
\hat x_a = \frac{\mathcal R_a}{\mathcal S} .
```

*Each coordinate from both sheets that carry it.* The spikes are pooled before the ratio:

```math
\hat x = \frac{\mathcal R^{(xy)}_x + \mathcal R^{(xz)}_x}{\mathcal S^{(xy)} + \mathcal S^{(xz)}} :
```

a weak packet counts in proportion to its spikes. Six ramps and three strengths become three
numbers.

*One population that holds a pair.* Per coordinate and per set of sheets, a pair population
holds

```math
\text{ramp} = \frac{\mathcal R_a}{1.5\,S_0}, \qquad \text{level} = \frac1{0.7}\Big(\frac{\mathcal S}{S_0} - 1\Big) ,
```

with $S_0$ the spikes per second of two packets as they should be. Its readout is fitted to

```math
\hat x_a = \frac{1.5\,\text{ramp}}{1 + 0.7\,\text{level}} ,
```

a smooth function of the pair on the range that occurs (A.5). This is the one division in
the network. It is cheap because level cells keep $\mathcal S$ near $S_0$, where the
function is nearly $1.5\,\text{ramp}(1 - 0.7\,\text{level})$.

*The comparison* is then a difference of two numbers per coordinate:

```math
d_i = \hat x_i^{\text{goal}} - \hat x_i^{\text{hand}} .
```

The goal sheets' readout is the "cold" side of B.6 and the forward sheets' the "warm" side.
Unlike B.6 the two ramps do not drive a decision: the difference is kept with its size,
because the cascade needs how far and not only which way.

**In spikes.** Every spike of the sheet neuron at $c_j$ goes to all 160 neurons of the
readout of each of its two coordinates with

```math
\mathsf w_{ij} = \frac{\alpha_i}{\varrho}\;\epsilon_i\cdot\begin{pmatrix}c_{j,a}/(1.5\,S_0)\\ 1/(0.7\,S_0)\end{pmatrix}\frac1{e\,\tau_{30}}, \qquad \tau_{30} = 30\ \text{ms},\ \varrho = 1.3 .
```

A spike is a vote for a place and a count at once. The "−1" of the level is a constant
current. The synapse is slow because two packets send about a thousand spikes a second (one
product population alone about 8,000): a pulse lasts 82 ms and 88 overlap; with 5 ms it
would be 15.

**Built.** `ring_se3/nest/ik_calibrate.py`, `scripts/calibrate_ramp.py`. $S_0$ = 1,082
spikes per second. Fitted on three real sheets with level cells, anchored through gate
populations at 257 places on a grid, each also with 25 pA taken from and added to every
sheet neuron: 771 measurements; no pose of any arm.

**Measured** (`docs/ik/results/tables.md` section 2):

- on 80 places it was not fitted on: off by 0.59% of the reach (root mean square), largest
  2.09%; on the same sheets without level cells 6.26%, largest 46.53%;
- read from the spikes of one tick, a coordinate scatters by 2.7% of the reach from tick to
  tick; over ten ticks it is 1.1% from the place;
- while the anchored place travels at 0.3 reaches per second the packets are 0.06 s behind
  and the ramp readouts 0.11 s;
- weights: median 6.0 pA, largest 71 pA; 89 of 160 neurons fire at 56 Hz, 251 spikes per
  tick.

**To derive further.** The scatter per tick from the shot noise of a packet (about 25
spikes per tick per sheet over a packet six cells across) and of the readout population;
the error of the fitted ratio as a function of the spread of $\mathcal S$.

## D.12 Level cells

**What it is for.** A homeostatic loop on the sheet itself that holds a packet at one
strength, so that a ramp means a place.

**Mathematics.** By D.4 an error of the anchor's height is the same current at every neuron
of the sheet. Each sheet has a sum population, the *level cells*, that holds

```math
v = \operatorname{clip}\Big(\frac1{0.8}\Big(\frac{\mathcal S}{S_{\text{set}}} - 1\Big),\ \pm1\Big)
```

and sends every neuron of the sheet the same current against it:

```math
u = -g_L\,v .
```

A current that is the same everywhere has no slope, so the place of the packet is not
touched.

*A proportional loop.* Near its set point a packet answers a small uniform current with
$\mathcal S = \mathcal S_{\text{free}} + G\,u$. Substituting $u = -g_L(\mathcal S/S_{\text{set}} - 1)/0.8$:

```math
\mathcal S - S_{\text{set}} = \mathcal S_{\text{free}} - S_{\text{set}} - \frac{G\,g_L}{0.8\,S_{\text{set}}}(\mathcal S - S_{\text{set}})
\;\;\Longrightarrow\;\;
\mathcal S - S_{\text{set}} = \frac{\mathcal S_{\text{free}} - S_{\text{set}}}{1 + K}, \qquad K = \frac{G\,g_L}{0.8\,S_{\text{set}}} .
```

What the anchor's error does to the strength is divided by $1 + K$, and of a missing
current the cells make up the fraction $K/(1+K)$.

*Stability.* The synapse from the sheet onto the level cells is two lags of $\tau_L$ (A.2);
the anchor's synapse onto the sheet two lags of $\tau_a$. Taking only these four,

```math
\mathcal L(s) = \frac{K}{(1 + s\tau_L)^2(1 + s\tau_a)^2} .
```

At $s = \mathrm i\omega$ its phase is $-2[\arctan\omega\tau_L + \arctan\omega\tau_a]$. It
reaches $-\pi$ where the two arctangents sum to $\pi/2$, i.e. where
$\omega\tau_L\cdot\omega\tau_a = 1$, $\omega^2 = 1/(\tau_L\tau_a)$. There
$|1 + \mathrm i\omega\tau_L|^2 = 1 + \tau_L/\tau_a$ and $|1 + \mathrm i\omega\tau_a|^2 = 1 + \tau_a/\tau_L$,
so $|\mathcal L| = K/[(1 + \tau_L/\tau_a)(1 + \tau_a/\tau_L)]$. The loop is stable while this
is below 1:

```math
K < 2 + \frac{\tau_L}{\tau_a} + \frac{\tau_a}{\tau_L} .
```

**In spikes.** Every sheet spike goes to all 60 level cells with one weight (median 7.0 pA)
through a 100 ms synapse (a pulse lasts 272 ms); half of the cells are driven by the sheet's
spikes and half held down. At the set point they hold zero and fire: 30 of 60 at 64 Hz.
Every level-cell spike goes to all 2,304 sheet neurons with the weight
$-g_L\delta_j/(e\,\tau_{\text{anchor}})$, the same for every receiver (median 0.96 pA,
largest 7.5 pA), through the anchor's synapse type.

**Built.** `ring_se3/nest/sheet.py: connect_level`. $S_{\text{set}}$ = 500 spikes per
second, $g_L$ = 200 pA, $\tau_L$ = 100 ms, $\tau_a$ = 20 ms. On all six sheets of the
network with the IK layer; the forward network of D.1 to D.6 is measured without them.

**Measured** (`docs/ik/results/tables.md` section 2, `docs/ik/spikes/figures.json`). 30 pA
less takes a packet from 552 to 344 spikes a second, so $G\approx6.9$ per second per pA and
$K = 6.9\cdot200/400\approx3.5$, $1 + K\approx4.5$.

| | Without | With | Ratio | Model |
| --- | ---: | ---: | ---: | ---: |
| Standard deviation of the strength over 80 places | 120 | 24 | 5.0 | 4.5 |
| Change of the strength when 30 pA are taken away | 208 | 47 | 4.4 | 4.5 |
| Change of the current the cells add | – | 23 pA (−5 to +18) | – | $30K/(1+K)$ = 23 pA |
| Packets weak (below 300 spikes a second), of 240 | 33 | 0 | | |

| $g_L$, $\tau_L$ | $K$ | Stable below (model) | sd over the places | sd from tick to tick |
| --- | ---: | ---: | ---: | ---: |
| none | 0 | | 112 | 61 |
| 100 pA, 30 ms | 1.7 | 4.2 | 35 | 88 |
| 300 pA, 30 ms | 5.2 | 4.2: swings | 52 | 632 |
| 100 pA, 100 ms | 1.7 | 7.2 | 36 | 67 |
| **200 pA, 100 ms** (built) | 3.5 | 7.2 | 22 | 69 |
| 300 pA, 150 ms | 5.2 | 9.6 | 18 | 76 |

The one setting the model calls unstable is the one that swings.

**To derive further.** $G$ from the sheet dynamics (how a packet's rate answers a uniform
current: the logarithmic rate law of A.1 summed over the packet); the packet's own lag in
$\mathcal L(s)$, which lowers the bound.

## D.13 The layer as populations

**What it is for.** D.7 to D.11 as products and sums, with the scaling that keeps every
held value in $[-1,1]$.

**Mathematics.** Held are $2e$ and $4d$ (full at 0.5 rad and a quarter of a reach per
component), $l_k/L_k$, $c_k$, the two halves $h^\pm_k = [\pm x_k]_0^1$ with
$x_k = t_k/\bar t_k$. In these, D.8 reads

```math
x_k = \frac{1}{2N_k\bar t_k}\,a_k\cdot(2e_k) + \frac{w^2\rho_k}{4N_k\bar t_k}\,c_k\cdot(4d_k) ,
```

```math
2e_{k+1} = 2e_k - 2\bar t_k\,a_k(h^+_k - h^-_k), \qquad 4d_{k+1} = 4d_k - 4\rho_k\bar t_k\,c_k(h^+_k - h^-_k) .
```

Everything in front of a product is a constant, so it is a weight.

**In spikes.**

| Value | Held by | Fed with |
| --- | --- | --- |
| goal matrix; $2e$ | 14 + 18 products, 4 + 3 sums (C.7) | goal rings, decoded orientation |
| goal position as a packet | 3 goal gates, 3 goal sheets with bias cells and level cells (D.4, D.12) | three numbers set from outside |
| $\hat x^{\text{goal}}$, $\hat x^{\text{hand}}$ | 6 ramp readouts of 160 | the sheets' spikes |
| $4d$ | 3 sums | goal readout with +4, hand readout with −4 |
| $l_k/L_k$ | 3 sums per joint from the second on | link entries, link offsets as weights |
| $c_k$ | 6 products, 3 sums per joint | axis entries, lever |
| $a_k\cdot e_k$, $c_k\cdot d_k$ | 3 + 3 products per joint | axis, direction, what is left |
| share | two halves of 60 | those products, with the weights above |
| what is left | 3 + 3 sums and 3 + 3 products | what was left, and the halves |
| lead of the joint's goal ring | gain pair and goal ring (C.10) | the two halves |

From the third joint on a joint costs 18 products, 9 sums and 2 halves for a pose (4,260
neurons, against 1,500 for orientation alone), and 12, 6 and 2 for a position.

**Built.** `ring_se3/ik/wiring.py` (the list of steps; checked against the rule for 12
chains and three tasks), `ring_se3/nest/ik_network.py`.

**Measured** (4-joint TIAGo arm, position goal, 35,916 neurons, `docs/ik/spikes/census.md`):

| Kind | How many | Neurons | Fire | Spikes per tick, each |
| --- | ---: | ---: | ---: | ---: |
| goal gate | 3 | 60 | 30 | 107 |
| goal sheet / forward sheet | 3 / 3 | 2,304 | 43 / 42 | 27 / 25 |
| level cells | 6 | 60 | 30 | 95 |
| ramp readout | 6 | 160 | 89 | 251 |
| sum: the offset | 3 | 60 | 37 | 99 |
| product, comparison and cascade | 38 | 200 | 114 | 407 |
| share half | 8 | 60 | 33 | 116 |
| gain pair, one side | 8 | 200 | – | 54 |
| goal ring of a joint | 4 | 200 | 22 | 103 |

All together 39,518 spikes per tick. Six sheets are 13,824 of the neurons and send 156 of
those spikes. Comparisons in spikes with the arm held: offset within 1.36% to 1.74% of the
reach in the median, turn direction within 2.9° and 4.6°, every share above 0.05 rad with
the right sign; the share error does not grow along the arm.

## D.14 Closing the loop on a pose

**What it is for.** Gain, delay, and where the hand ends.

**Mathematics.** As in C.11, with $\dot q_k = \kappa t_k$ and D.8:

```math
\dot d(t) = -\kappa\,d(t - \tau), \qquad \dot\Delta(t) = -\kappa\,\Delta(t - \tau) ,
```

stable for $\kappa\tau < \pi/2$ and without overshoot for $\kappa\tau\le1/e$. In between,
the roots of $s = -\kappa e^{-s\tau}$ are complex, $s\tau = -\sigma\pm\mathrm i\omega$, and
the first overshoot is $e^{-\sigma\pi/\omega}$ of the distance: for $\kappa\tau = 0.55$,
$s\tau = -0.73\pm0.88\,\mathrm i$ and about 7%.

*The delay of the position*, from parts measured on their own: the forward sheets behind a
moving arm 0.21 to 0.25 s (D.6); a ramp readout behind its packet 0.05 s (D.11); the lead of
a goal ring behind its share 0.20 to 0.30 s (C.10). Together about half a second, so with
$\kappa$ = 1 per second $\kappa\tau\approx0.5$ to 0.6.

*Where the hand ends.* The loop makes the *decoded* turn and offset zero. The hand comes to
rest where the network believes the goal to be; the true error that remains is the sum of
how far the forward network's pose is from the truth, how far the goal's packets are from
the goal, and what the readouts and the comparison add. No gain makes the hand more accurate
than the forward network is.

*A goal that travels.* Lift, pitch, yaw at up to 1 rad/s; the position on a straight line
at up to 0.3 reaches per second. An anchor that travels carries its packet, one that jumps
silences it (D.4); and the offset population is full at a quarter of a reach.

*Where the rule stops short.* $J^\top E = 0$ with $E\ne0$: a pose from which no joint moves
the hand toward the goal (a stretched arm with the goal further along it; a joint at its
limit that alone could help). The rule follows $\lVert E\rVert$ downhill and does not plan.

**In spikes.** A trial ends when the written goal has arrived and the hand has turned less
than 0.5° and moved less than 0.4% of the reach within 2 s, or after 25 s.

**Built.** `ring_se3/ik/session.py`, `ring_se3/ik/experiments.py`; $\kappa$ = 1 per second,
$w$ = 3.

**Measured** (spiking network, test pairs, marks fixed beforehand;
`docs/ik/results/tables.md` sections 0, 4, 5):

| Arm, task | Trials | Reached | Final error, median | Within the marks after | Path / straight |
| --- | ---: | ---: | --- | ---: | ---: |
| TIAGo, 4 joints, position | 40 | 98% | 1.96% of the reach | 4.5 s | 1.36 |
| generic, 4 joints, position | 40 | 100% | 1.24% | 4.0 s | 1.32 |
| generic, 7 joints, pose | 30 | 97% | 2.02%, 2.27° | 5.0 s | 1.54 (mark 1.5 missed) |
| TIAGo, 7 joints, pose | 30 | 93% | 2.09%, 1.89° | 5.7 s | 1.43 |

- During these trials the forward sheets are off by 1.26% to 1.68% of the reach and the
  decoded orientation by 0.99° to 1.61°: the floor. The exact rule with a readout off by 2°
  and 2.6% of the reach ends 2.04° to 2.12° and 2.63% to 2.64% from the goal.
- Positions compared as numbers and not on the sheets: 1.62% against 1.96% (4 joints), 1.70%
  against 2.09% (7 joints).
- Every joint getting the whole error (4 joints): 88% reached, after 8.1 s.
- Exact rule: gain 0.75 gives 6.3 to 6.8 s, gain 1.5 gives 3.5 to 4.0 s with a path of 1.80
  to 1.95 times the straight line; goals anywhere in the joint ranges 37% to 77% reached.
- Speed with 12 threads on an idle machine: 0.66 of real time (4 joints), 0.22 (7 joints).

**To derive further.** The delay equation with rate limits; the noise floor of the hand at
rest from the per-tick scatter of D.11 filtered by the loop.

## D.15 Gazebo for the pose

**What it is for.** The same loop with the TIAGo arm's own controller and gravity (C.12),
and a check that the "true position" computed from joint angles is where the simulated arm
is.

**Built.** `ring_so3/robots.py`, `ring_se3/gazebo/link.py`, `ring_se3/gazebo/robot.py`,
`scripts/run_ik_gazebo.py`. Two things found on the way: Gazebo's service multiplies two
orientations in the wrong order when asked for a relative pose, so both poses are taken in
the world frame and composed; and a wrist joint reports a velocity while standing still, so
"at rest" is read from the angles.

**Measured** (`docs/ik/results/tables.md` section 9):

- end point from Gazebo's link states against the chain's kinematics at 5 held poses: 0.22
  mm at most;
- TIAGo, 7 joints, pose, 30 pairs: ring IK 93% reached (all 28 of the goals it reaches on
  the light robot), 2.22% of the reach and 1.50°, 5.8 s; damped least squares with exact
  joint angles 100%, 0.29% and 0.35°;
- TIAGo, 4 joints, position, 20 pairs: 100% reached, 1.74%;
- a joint is 2.30° behind the target it was sent in the median (7 joints).

---

# Part E. How the three fit together

## E.1 From the mathematics to the spikes: one table

Spike counts are per 50 ms tick, from the census of each repo (SO-3: a 3-joint chain with
its IK layer; SE-3: the 4-joint TIAGo arm with a position goal).

| Mathematics | Part | Neurons that carry it | What makes them fire | Measured |
| --- | --- | --- | --- | --- |
| joint angle $q_k$ | B.1 to B.3 | a ring of 200 ring neurons | random generator spikes onto 11 neurons; the ring's own spikes around them | 27 fire; 51 spikes |
| $(\cos q_k, \sin q_k)$ | C.3 | 120 neurons, directions all round | ring spikes weighted by the cos and sin of the sender's angle | 71 fire; 287 |
| a product $ab$ | A.6 | 200 neurons in four groups | the spikes that carry $a$ and $b$; each group sees one of $\pm a\pm b$ | 110 fire; 447 |
| an entry of $R_k$ | C.2 | 60 neurons | spikes of two product populations, added in the membrane | 30 fire; 148 |
| lift, pitch, yaw as bumps | C.4 | 3 output rings of 100 | a pair weighted by $\cos\theta_j$, $\sin\theta_j$, above a threshold | 34 fire; 43 to 228 |
| goal orientation $G$ | C.7 | 3 goal rings; 14 products, 4 sums | as a joint ring; the wiring of a z, y, x chain | 469; 151 |
| turn $e = \tfrac12\sum r_i\times g_i$ | C.7 | 18 products, 3 sums | spikes that carry entries of $R$ and of $G$ | 437; 132 (at the goal 93) |
| position $u$ as numbers | D.1 | 3 position cells of 60 | spikes that carry link entries, link offsets as weights | 30 fire; 106 |
| position as a packet | D.3, D.4 | 3 sheets of 2,304 | gate spikes weighted by the receiver's place, bias cells, the sheet's own spikes | 42 fire; 25 |
| goal position as a packet | D.4 | 3 goal gates, 3 goal sheets | the same, from three numbers set from outside | 43 fire; 27 |
| strength of a packet | D.12 | 60 level cells per sheet | every sheet spike with one weight, through 100 ms | 30 fire; 95 |
| a coordinate $\hat x_a$ | D.11 | a ramp readout of 160 | spikes of two sheets, weighted by the sender's place and counted | 89 fire; 251 |
| offset $d$ | D.11 | 3 sums of 60 | two ramp readouts, goal with +, hand with − | 37 fire; 99 (at the goal 94) |
| lever, direction $c_k$ | D.7 | 3 sums; 6 products and 3 sums per joint | link entries; axis and lever | 110; 407, 110 |
| $a_k\cdot e_k$, $c_k\cdot d_k$ | C.8, D.8 | 3 (and 3) products per joint | axis or direction, and what is left | 407 to 446 |
| share $t_k$ | C.9, D.13 | two half populations of 60 | those products; near a limit the joint's ring, negatively | 116 to 261 (at zero 93 to 106) |
| what is left | C.8, D.8 | 3 (and 3) sums and products per joint | what was left, and the halves | 95 to 110 |
| $r_R$, $r_L$ | C.10 | gain pair, 2 x 200 gain neurons | state-ring spikes and a constant current together; the share moves the sides apart | 54 per side at rest; 84 and 13 at a full share |
| lead $\ell$, the joint's next angle | C.10 | the joint's goal ring, 200 | gain spikes 8 neurons up and 8 down | 22 fire; 103 to 105 |
| SO(2) only: ramp, decision | B.6 | 40 feature neurons, 2 ramp neurons, 2 decision neurons | weighted ring spikes; mutual inhibition | – |
| SO(2) only: moving a bump | B.7 | gain pair, 2 x 200 | ring spike AND decision; inhibits the neighbour | – |

## E.2 What is reused, and what each stage adds

| Element | SO(2) | SO(3) | SE(3) |
| --- | --- | --- | --- |
| ring attractor, its neuron, the wall | introduced | reused for joint and goal rings | reused; the sheet is its 2-D form |
| encoder | 11 neurons at 200 Hz; once or corrective | every tick, between indices | the same |
| population vector | introduced | reused for rings | centre of spikes on a sheet |
| ramp comparator | introduced (far) | not used | returns as the ramp readout, graded |
| projection comparison | introduced (near) | not used | not used |
| gain pair | moves an attractor's bump | writes a goal ring beside the state bump | the same |
| predictive ring, body pair | introduced (v4) | replaced: the loop closes through the forward network | the same |
| decoder to velocity | introduced | replaced: the joint follows the goal ring's angle | the same |
| value, product and sum populations | – | introduced | reused |
| orientation, joint by joint | – | introduced | reused unchanged |
| comparison $e$, cascade, share halves, hold at a limit | – | introduced | reused; the share gains the offset part and a divisor |
| position cells, sheets, anchor, velocity cells | – | – | introduced |
| level cells, ramp readout, goal sheets | – | – | introduced |

Two ideas run through all three:

- **A comparison of two place codes becomes a signed drive.** SO(2): two ramps and a
  decision, then a projection. SO(3): a cross product of matrix columns. SE(3): a difference
  of two ramp readouts.
- **The amplitude of an attractor's activity must not be mistaken for its place.** SO(2):
  the corrective encoder. SO(3): readouts fitted on real bumps. SE(3): level cells, and a
  readout of ramp over strength.

## E.3 Sizes and speed

| Network | Neurons | Source |
| --- | ---: | --- |
| SO(3) forward, 2 joints | 1,860 | SO-3 `docs/results.md` |
| SO(3) with its IK layer, 3 joints | 16,900 | SO-3 `TECHNICAL.md`, section 28 |
| SO(3) with its IK layer, 7-joint TIAGo arm | 37,620 | SO-3 `README.md` |
| SE(3) forward, 2 joints / 7 joints | 11,276 / 36,356 | SE-3 `docs/results.md` |
| SE(3) with the IK layer, 4 joints, position | 35,916 (IK layer 20,948) | SE-3 `docs/ik/results/tables.md` |
| SE(3) with the IK layer, 7-joint TIAGo / generic, pose | 69,036 / 71,976 | the same |

- Products: 62 for the orientation of 7 joints (166 if computed all at once); 128 in the
  pose IK layer of the 7-joint TIAGo arm.
- Speed: the SE(3) forward network of 7 joints runs at 0.44 of real time on 8 threads; with
  the IK layer at 0.22 on 12 threads, the 4-joint one at 0.66.

## E.4 What is shown and what is not

| Claim | Stage | Evidence | Limit |
| --- | --- | --- | --- |
| A spiking ring controller takes a joint to its goal to about one ring index | SO(2) | v4: 0.008 / 0.022 / 0.045 rad over seven joints in Gazebo, 100% settled | half a ring index; repeats to 0.0133 rad |
| The result does not depend on the actuator's gain | SO(2) | v4 at 0.7 and 1.3 of the motor gain: 0.013 and 0.010 rad | v3 does depend on it |
| The rings decode to the orientation of every link | SO(3) | 0.43° (2 joints) to 1.14° (7 joints) in the median over 300 poses per model | 0.16 to 0.20 s behind a moving arm; not robust to losing product neurons |
| A spiking network turns the hand to a goal orientation in closed loop | SO(3) | 100% of 67 and 94% of 83 goals; 1.32° at the end | light robot and Gazebo; goals reachable |
| The rings decode to the position of the end point, held on sheets | SE(3) | 0.97% to 1.37% of the reach in the median, 12 models | the sheets hold the position, they do not sharpen it; a free packet drifts |
| A position on sheets can be compared as a number | SE(3) | ramp readout off by 0.59% of the reach with level cells | 6.26% without; one tick scatters by 2.7% |
| A spiking network takes the hand to a pose goal, without a solver | SE(3) | 93% to 100% of 30 to 40 goals per arm; about 2% of the reach and 2° at the end | goals near the start; no more exact than the forward readout; 30 to 40 pairs, one seed |
| It works with a simulated robot that has dynamics | SE(3) | 93% of 30 and 100% of 20 goals in Gazebo | simulation only |

Not shown by anything in the three repos: that the ring IK is more exact or faster than a
standard method (damped least squares with exact joint angles is better on both); that
goals anywhere in the workspace are reached (the rule itself reaches 37% to 77%); that path
integration carries a packet; that the networks tolerate the loss of neurons; that the arm's
geometry need not be known (link offsets and joint axes are weights); that any of it carries
over to a real robot.

---

# Back matter

## F.1 Notation of this document against the three references

| Here | Meaning | SO-2 reference | SO-3 reference | SE-3 reference |
| --- | --- | --- | --- | --- |
| $\rho_q$ | radians per ring index | $\rho$ | $\rho_k$ (section 21) | – |
| $\varrho$ | radius of a population's range | – | $\rho$ | $\rho$ (section 34) |
| $\rho_k$ | largest lever of joint $k$ over the reach | – | – | $\rho_k$ |
| $\zeta$ | room before a joint limit | – | $\rho$ (section 19) | – |
| $\epsilon_i$ | encoder of a population neuron | – | $e_i$ | $e_i$ |
| $\delta_j$ | decoder | – | $d_j$ | $d_j$, $d^{(1)}$, $d^{(2)}$ |
| $\eta$ | scale from ring spikes to a pair, $1/B_1$ | – | $\kappa$ (section 9) | – |
| $\kappa$ | loop gain | – | $\kappa$ (section 22) | $\kappa$ |
| $\chi$ | shift of a sheet neuron's outgoing weights, cells | – | – | $\kappa$ (section 5) |
| $\lambda_s$ | profile units per cell | – | – | $\lambda$ |
| $\ell_c$, $E_s$, $n_s$ | cell, half-width, side of a sheet | – | – | $\delta$, $E$, $n$ |
| $g_a$, $b_a$ | gain and top of the anchor | – | – | $g$, $b$ |
| $g_o$, $h$ | gain and threshold of an output ring | – | $g$, $h$ | – |
| $\bar t_k$ | largest share | – | $c_k$ | $\text{cap}_k$ |
| $c_k$ | direction in which joint $k$ moves the end point | – | – | $c_k$ |
| $c^o_k$ | origin of joint $k$ | – | – | $c_k$ (section 2) |
| $e$, $\hat n$ | the turn; its axis | – | $e$, $n$ | $e$, $n$ |
| $\mu_b$, $s_0$ | bump displacement per spike, dead zone | $a$, $\theta$ | – | – |
| $n_R$, $n_L$ | gain spikes per tick | $R$, $L$ | – | – |
| $\Gamma$, $\nu_k$ | ramp, its weights | $g$, $a_k$ | – | – |
| $\pi$, $\sigma_s$ | predictive and state bump | $p$, $s$ | – | – |
| $\varepsilon$, $\eta$ (B.10 only) | body's error, prediction ahead of body | $e$, $d$ | – | – |
| $\lambda_s$ (B.8 only) | filter factor of the decoder | $\alpha$ | – | – |
| $\beta_m$ | joint motion per tick per index of offset | $\beta$ | – | – |
| $K_t$ | NEST steps per tick | $K$ | – | – |
| $K$ | gain of the level loop | – | – | $K$ |
| $\Omega$ | angular speed of the hand | – | $\omega$ | – |
| $\omega_k$ | relaxation factor | – | – | $\omega_k$ |

Two symbols are used twice here, each time locally: $\eta$ (C.3 and B.10) and $\lambda_s$
(B.8 and D.3).

## F.2 Where the numbers come from

Every measured number in this document was copied from one of these files; none was
measured again for it.

| Part | Source |
| --- | --- |
| A, B.2 (spikes of a neuron, of a bump, weights) | SO-3 `TECHNICAL.md` part III; SO-3 `docs/spikes/census.md` |
| B (all else) | SO-2 `TECHNICAL.md`; SO-2 `docs/results.md` |
| C.1 to C.6 | SO-3 `TECHNICAL.md` part I; SO-3 `docs/results.md` |
| C.7 to C.13 | SO-3 `TECHNICAL.md` parts II, III and appendix A; SO-3 `docs/ik/results/tables.md` |
| D.1 to D.6 | SE-3 `TECHNICAL.md` part I; SE-3 `docs/results.md`, `docs/results/tables.md` |
| D.7 to D.15 | SE-3 `TECHNICAL.md` parts II, III; SE-3 `docs/ik/results/tables.md`, `docs/ik/spikes/census.md`, `docs/ik/spikes/figures.json` |
| E | the above |

State of the repos when this was written (2026-10-12): SO-2 on branch
`velocity-calibration`, SO-3 on `ik-cascade`, SE-3 on `ik-pose`.
