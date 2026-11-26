"""The training and evaluation protocol of the learned baselines, and their shared command line.

``python -m <model> COMMAND``, with ``<model>`` ``mpc_nn_dynamics`` or ``pets``::

    collect     phase 1: random-action episodes on training starts (results/<model>/training/transitions.npz)
    train       fit the model on every transition so far; with --iterations K, then K rounds of phase 2:
                MPC episodes on training goals with the current model, add their transitions, refit
    reach       the final model's MPC on the 30 reported pairs (results/<model>/reach_gazebo/ or reach_kinematic/)
    summarise   recompute the reaching summary
    all         collect, train with the configured iterations, reach

Options: ``--plant gazebo|light`` (default gazebo: run beside ``docker/run.bash gazebo up NAME --image baselines --cuda``),
``--device auto|cuda|cpu``, ``--seed`` (default 1: model initialisation, minibatches, random actions, planner sampling),
``--pairs`` (default 30), ``--max-minutes``.

Every step resumes: collection counts the episodes already stored, training reads ``training.json`` for the iterations
done, and reaching resumes its run file. ``training.json`` records the configuration, the device and versions, every
iteration's transitions, fit and wall time.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional

import numpy as np

from reaching_benchmark import results, summary
from reaching_benchmark.benchmark import BENCHMARKS, PLANTS, make_plant, run_reach, test_pairs
from reaching_benchmark.geometry.chains import get_chain
from reaching_benchmark.progress import Progress, say
from reaching_benchmark.targets import TRAIN_SEED, pose_pairs
from reaching_benchmark.task import IkDesign, default_task

ARM = "tiago7"


@dataclasses.dataclass(frozen=True)
class Config:
    """One learned baseline's settings: the model, its training and its planner, and the protocol's budget."""

    name: str
    label: str
    reference: str
    # the model
    members: int
    probabilistic: bool
    hidden: int
    layers: int
    activation: str
    # its training
    learning_rate: float
    weight_decay: float
    batch: int
    epochs: int
    # the planner
    planner: str                      # "random_shooting" or "cem"
    horizon: int
    planner_settings: Dict[str, Any]
    # the protocol
    random_episodes: int = 20
    random_episode_s: float = 10.0
    iterations: int = 10
    episodes_per_iteration: int = 3

    def describe(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)


def training_folder(config: Config) -> str:
    return os.path.join(results.ROOT, config.name, "training")


def paths(config: Config) -> Dict[str, str]:
    folder = training_folder(config)
    return {"transitions": os.path.join(folder, "transitions.npz"), "model": os.path.join(folder, "model.pt"),
            "record": os.path.join(folder, "training.json")}


def read_record(config: Config) -> Dict[str, Any]:
    path = paths(config)["record"]
    if not os.path.exists(path):
        return {"model": config.name, "config": config.describe(), "phase1": None, "iterations": [], "wall_s": 0.0}
    with open(path, encoding="utf-8") as stream:
        return json.load(stream)


def write_record(config: Config, record: Dict[str, Any]) -> None:
    results.write(paths(config)["record"], record, digits=6, pretty=True)


def controller_description(config: Config, record: Dict[str, Any]) -> Dict[str, Any]:
    return {"name": config.label, "reference": config.reference,
            "reads": "the measured joint angles and velocities, every 50 ms tick",
            "given": "the true geometry for the cost (forward kinematics of predicted joint angles)",
            "learns": "the arm's response to joint steps, from %s transitions" % record.get("transitions", "?"),
            "internal_model": "learned dynamics, planned over %d ticks ahead (predictive)" % config.horizon,
            "planner": config.planner, "config": config.describe(), "device": record.get("device")}


# ---- building the model and the planner --------------------------------------------------------------------------
def build_model(config: Config, chain: Any, device: Any):
    from .model import DynamicsModel
    return DynamicsModel(2 * chain.n, chain.n, config.hidden, config.layers, config.activation, config.members,
                         config.probabilistic).to(device)


def build_planner(config: Config, model: Any, chain: Any, design: IkDesign, device: Any, seed: int):
    from reaching_benchmark.limits import step_caps
    from .kinematics import TorchChain
    from .planner import CrossEntropy, RandomShooting
    cls = {"random_shooting": RandomShooting, "cem": CrossEntropy}[config.planner]
    return cls(model, TorchChain(chain, device), design, step_caps(chain, design), config.horizon, seed, **config.planner_settings)


def load_model(config: Config, chain: Any, device: Any):
    import torch
    model = build_model(config, chain, device)
    model.load_state_dict(torch.load(paths(config)["model"], map_location=device, weights_only=True))
    model.eval()
    return model


# ---- the steps ------------------------------------------------------------------------------------------------------
def design_for(chain: Any, seconds: Optional[float] = None) -> IkDesign:
    design = IkDesign(task=default_task(chain.n), kappa=1.0)
    return dataclasses.replace(design, trial_s=float(seconds)) if seconds else design


def episode(loop: Any, pair: Any, settle: bool = True) -> Any:
    if hasattr(loop.robot, "link"):
        loop.robot.link.put_base_back()
    loop.place(pair.start)
    return loop.reach(pair.goal, pair.goal_position, stop_when_settled=settle)


def collect(config: Config, plant: str, seed: int) -> None:
    """Phase 1: ``random_episodes`` episodes of ``random_episode_s`` from training starts, every step random."""

    from .controller import RandomActionLoop
    from .data import Transitions
    chain = get_chain(ARM)
    data = Transitions.load(paths(config)["transitions"])
    done = data.episode_count()
    if done >= config.random_episodes:
        say("%s: phase 1 done already (%d episodes, %d transitions)" % (config.name, done, len(data)))
        return
    record = read_record(config)
    started = time.time()
    pairs = pose_pairs(chain, config.random_episodes, TRAIN_SEED)
    loop = RandomActionLoop(chain, design_for(chain, config.random_episode_s), make_plant(plant, chain), data, seed)
    loop.warm_up(20)
    progress = Progress("%s random episodes" % config.name, config.random_episodes, done, record.get("wall_s", 0.0), unit="episode")
    for number in range(done, config.random_episodes):
        loop.episode = number
        loop.rng = np.random.default_rng([int(seed), number])
        progress.start("%d" % number)
        episode(loop, pairs[number], settle=False)
        data.save(paths(config)["transitions"])
        progress.finish(None, "%s random episode %d: %d transitions in all" % (config.name, number, len(data)))
    progress.close()
    record.update(phase1={"episodes": config.random_episodes, "episode_s": config.random_episode_s, "plant": plant, "seed": seed},
                  transitions=len(data), wall_s=record.get("wall_s", 0.0) + time.time() - started)
    write_record(config, record)


def fit_and_save(config: Config, chain: Any, device: Any, device_info: Dict[str, Any], seed: int, iteration: int) -> Dict[str, Any]:
    import torch
    from .data import Transitions
    from .train import fit
    data = Transitions.load(paths(config)["transitions"])
    if not len(data):
        raise SystemExit("%s: no transitions yet: run 'collect' first" % config.name)
    torch.manual_seed(int(seed) * 1000 + iteration)
    model = build_model(config, chain, device)
    stats = fit(model, data, config.epochs, config.batch, config.learning_rate, config.weight_decay, int(seed) * 1000 + iteration, device)
    os.makedirs(training_folder(config), exist_ok=True)
    torch.save(model.state_dict(), paths(config)["model"])
    stats.update(episodes=data.episode_count(), device=device_info)
    return stats


def train(config: Config, plant: str, seed: int, iterations: int, device_name: str) -> None:
    """Fit on everything so far; then ``iterations`` rounds of phase 2 (resumes after the rounds already done)."""

    from .controller import MpcLoop
    from .data import Transitions
    from .setup import setup
    chain = get_chain(ARM)
    device, info = setup(seed, device_name)
    record = read_record(config)
    record["device"] = info
    started = time.time()
    if not record["iterations"]:
        say("%s: fitting on phase-1 data" % config.name)
        record["iterations"].append({"iteration": 0, "fit": fit_and_save(config, chain, device, info, seed, 0)})
        write_record(config, record)
    pairs = pose_pairs(chain, config.random_episodes + iterations * config.episodes_per_iteration, TRAIN_SEED)
    design = design_for(chain)
    progress = Progress("%s iterations" % config.name, iterations, len(record["iterations"]) - 1, unit="iteration")
    while len(record["iterations"]) - 1 < iterations:
        number = len(record["iterations"])
        progress.start("%d" % number)
        data = Transitions.load(paths(config)["transitions"])
        model = load_model(config, chain, device)
        loop = MpcLoop(chain, design, make_plant(plant, chain), build_planner(config, model, chain, design, device, seed * 1000 + number), data)
        loop.warm_up(20)
        reached = []
        for k in range(config.episodes_per_iteration):
            pair = pairs[config.random_episodes + (number - 1) * config.episodes_per_iteration + k]
            loop.episode = data.episode_count()
            trial = episode(loop, pair)
            reached.append(bool(trial.reached(design)))
            data.save(paths(config)["transitions"])
        entry = {"iteration": number, "training_pairs": [int(p.index) for p in pairs[config.random_episodes + (number - 1) * config.episodes_per_iteration:
                                                                                   config.random_episodes + number * config.episodes_per_iteration]],
                 "reached_on_training_goals": reached, "fit": fit_and_save(config, chain, device, info, seed, number)}
        record["iterations"].append(entry)
        record["transitions"] = entry["fit"]["transitions"]
        record["wall_s"] = record.get("wall_s", 0.0) + time.time() - started
        started = time.time()
        write_record(config, record)
        progress.finish(None, "%s iteration %d: %d transitions, reached %d/%d training goals" % (
            config.name, number, record["transitions"], sum(reached), len(reached)))
    progress.close()
    record["wall_s"] = record.get("wall_s", 0.0) + time.time() - started
    write_record(config, record)


def reach(config: Config, plant: str, seed: int, pairs: int, device_name: str, max_minutes: Optional[float]) -> None:
    from .controller import MpcLoop
    from .setup import setup
    chain = get_chain(ARM)
    device, info = setup(seed, device_name)
    record = read_record(config)
    model = load_model(config, chain, device)
    design = design_for(chain)
    loop = MpcLoop(chain, design, make_plant(plant, chain), build_planner(config, model, chain, design, device, seed * 1000 + 999))
    run = {"arm": ARM, "pairs": pairs, "kappa": design.kappa, "plant": plant, "kind": "near", "seed": seed,
           "iterations": len(record["iterations"]) - 1, "transitions": record.get("transitions"), "device": info}
    name = results.run_name(ARM, seed, task=design.task)
    run_reach(config.name, BENCHMARKS[plant], name, run, loop, test_pairs(chain, pairs), controller_description(config, record),
              max_minutes=max_minutes)


def main(config: Config, argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m %s" % config.name, description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("collect", "train", "reach", "summarise", "all"))
    parser.add_argument("--plant", choices=PLANTS, default="gazebo")
    parser.add_argument("--device", default="auto", help="auto (the GPU when there is one), cuda or cpu")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=None, help="rounds of phase 2 (default: the model's own, %d)" % config.iterations)
    parser.add_argument("--pairs", type=int, default=30)
    parser.add_argument("--max-minutes", type=float, default=None)
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)
    iterations = config.iterations if args.iterations is None else args.iterations
    if args.command in ("collect", "all"):
        collect(config, args.plant, args.seed)
    if args.command in ("train", "all"):
        train(config, args.plant, args.seed, iterations, args.device)
    if args.command in ("reach", "all"):
        reach(config, args.plant, args.seed, args.pairs, args.device, args.max_minutes)
    if args.command in ("reach", "summarise", "all"):
        benchmark = BENCHMARKS[args.plant]
        print("wrote", results.write_summary(config.name, benchmark, summary.summarise(config.name, benchmark)))
    return 0
