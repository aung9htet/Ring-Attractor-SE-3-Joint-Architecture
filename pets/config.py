"""Settings of PETS, after Chua et al. (2018): an ensemble of 5 probabilistic networks (four hidden layers of 200 swish
units, Gaussian outputs with learned bounds on the log variance, negative log-likelihood, each member on its own bootstrap
of the data); CEM with a population of 400, 40 elites, 5 iterations and smoothing 0.1 over a 25-step horizon, each
sequence scored by 20 particles under TS∞; model-based RL iterations that aggregate on-policy data and retrain."""

from learned_dynamics.protocol import Config

CONFIG = Config(
    name="pets", label="PETS", reference="Chua et al., NeurIPS 2018",
    members=5, probabilistic=True, hidden=200, layers=4, activation="swish",
    learning_rate=1e-3, weight_decay=5e-5, batch=256, epochs=25,
    planner="cem", horizon=25, planner_settings={"population": 400, "elites": 40, "iterations": 5, "alpha": 0.1, "particles": 20},
)
