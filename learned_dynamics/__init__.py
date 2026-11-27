"""What the two learned baselines share (NN dynamics + MPC, :mod:`mpc_nn_dynamics`; PETS, :mod:`pets`), so that they
differ only where their papers differ: the model (one deterministic network, or an ensemble of probabilistic ones) and the
planner (random shooting, or CEM with trajectory sampling).

**The protocol** (:mod:`learned_dynamics.protocol`), identical for both:

* **State** ``s = (q, q̇)``: the measured joint angles and velocities (Gazebo's joint states).
* **Action** ``u``: a joint step with ``|u_k| ≤ cap_k``, the same caps as every controller
  (:func:`reaching_benchmark.limits.step_caps`), sent to the plant as the target ``q + u / κ``, exactly as DLS IK's step.
* **What is learned**: how the arm responds, ``Δs = f(s, u)``. The geometry is supplied: the planner scores each predicted
  ``q`` by forward kinematics against the written goal (:mod:`learned_dynamics.kinematics`), as the ring-attractor
  kinematics model and DLS IK are given the geometry. Nothing about the task is learned.
* **Data**: training starts and goals come from ``TRAIN_SEED``; the 30 reported pairs are never seen in training.
  Phase 1: random-action episodes. Phase 2: iterations of model-based RL as in both papers: run the current model's MPC on
  training goals, add the transitions, retrain on everything.
* **Evaluation**: the final model's MPC on the 30 reported pairs through the shared runner
  (:func:`reaching_benchmark.benchmark.run_reach`), without further learning.

Everything runs in PyTorch on the GPU when there is one (``docker/run.bash ... --image baselines --cuda``), with
deterministic algorithms and fixed seeds (:mod:`learned_dynamics.setup`); the device and versions go into every file.
"""
