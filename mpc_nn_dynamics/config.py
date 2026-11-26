"""Settings of NN dynamics + MPC, after Nagabandi et al. (2018): one network with two hidden layers of 500 ReLU units
predicting the normalised state change, trained by mean squared error with Adam; random-shooting MPC over a 10-step
horizon with 1,000 uniformly drawn action sequences, the first action carried out, replanned every tick; model-based RL
iterations that aggregate on-policy data and retrain."""

from learned_dynamics.protocol import Config

CONFIG = Config(
    name="mpc_nn_dynamics", label="NN dynamics + MPC", reference="Nagabandi et al., ICRA 2018",
    members=1, probabilistic=False, hidden=500, layers=2, activation="relu",
    learning_rate=1e-3, weight_decay=0.0, batch=512, epochs=60,
    planner="random_shooting", horizon=10, planner_settings={"samples": 1000},
)
