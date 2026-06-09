"""NEAT, evaluated on the shared CarEnv so it appears on the same charts.

This is the incumbent technique. It reuses the project's existing config.txt
(via play.build_config) and neat-python, but drives carenv.CarEnv instead of
play.py's pygame-drawing loop, so its env-step count and reward are measured the
exact same way as GA / ES / DQN.

NEAT has no notion of a step budget, so we run it one generation at a time and
stop once the shared budget is spent.

Trainer contract matches the others:
    train(env, budget_steps, seed, record) -> (best_fitness, best_genome)
"""

import random

import neat
import numpy as np

import play
from carenv import rollout


def train(env, budget_steps, seed, record):
    random.seed(seed)  # neat-python draws from the global random module
    config = play.build_config()
    pop = neat.Population(config)

    state = {"best_fitness": -np.inf, "best_genome": None}

    def eval_genomes(genomes, cfg):
        gen_best = -np.inf
        gen_info = {"laps": 0}
        for _, g in genomes:
            net = neat.nn.FeedForwardNetwork.create(g, cfg)

            def act(obs, net=net):
                return int(np.argmax(net.activate(obs)))

            fitness, info = rollout(env, act)
            g.fitness = fitness
            if fitness > gen_best:
                gen_best, gen_info = fitness, info
            if fitness > state["best_fitness"]:
                state["best_fitness"] = fitness
                state["best_genome"] = g
        record(float(gen_best), {"laps": gen_info.get("laps", 0)})

    # One generation per iteration until the env-step budget is exhausted.
    while env.total_steps < budget_steps:
        pop.run(eval_genomes, 1)

    return state["best_fitness"], state["best_genome"]
