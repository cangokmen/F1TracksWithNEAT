"""Fixed-topology Genetic Algorithm.

The direct counterpart to NEAT: same evolutionary paradigm (a population scored
by episode return, survival of the fittest, mutation), but the network topology
is FIXED (techniques/mlp.py). Only the weights evolve. Comparing this to NEAT
isolates the question "does growing topology actually buy anything here?"

Search operators:
  - tournament selection
  - uniform (per-gene) crossover
  - Gaussian mutation
  - elitism (best few carried over unchanged)

Trainer contract (shared by all techniques):
    train(env, budget_steps, seed, record) -> (best_fitness, best_params)
  where record(fitness, info) is called once per generation with that
  generation's best episode return. The benchmark stamps env.total_steps and
  wall-clock onto each record() call, so trainers stay ignorant of timing.
"""

import numpy as np

from carenv import NUM_INPUTS, NUM_OUTPUTS, rollout
from techniques.mlp import MLPPolicy, param_count

POP = 40
ELITE = 4
TOURNAMENT = 3
INIT_STD = 1.0
MUT_RATE = 0.15      # probability each gene is perturbed
MUT_SIGMA = 0.4      # std of the Gaussian perturbation


def _evaluate(env, genome):
    policy = MLPPolicy(genome, NUM_INPUTS, NUM_OUTPUTS)
    fitness, info = rollout(env, policy.act)
    return fitness, info


def _tournament(rng, fitnesses):
    contenders = rng.integers(0, len(fitnesses), size=TOURNAMENT)
    return contenders[np.argmax(fitnesses[contenders])]


def _crossover(rng, a, b):
    mask = rng.random(a.shape) < 0.5
    return np.where(mask, a, b)


def _mutate(rng, genome):
    mask = rng.random(genome.shape) < MUT_RATE
    noise = rng.normal(0.0, MUT_SIGMA, size=genome.shape) * mask
    return genome + noise


def train(env, budget_steps, seed, record):
    rng = np.random.default_rng(seed)
    dim = param_count(NUM_INPUTS, 12, NUM_OUTPUTS)
    pop = rng.normal(0.0, INIT_STD, size=(POP, dim)).astype(np.float32)

    best_fitness = -np.inf
    best_params = pop[0].copy()

    while env.total_steps < budget_steps:
        fitnesses = np.empty(POP, dtype=np.float64)
        gen_best_info = None
        for i in range(POP):
            fitnesses[i], info = _evaluate(env, pop[i])
            if fitnesses[i] > best_fitness:
                best_fitness = fitnesses[i]
                best_params = pop[i].copy()
                gen_best_info = info
            if env.total_steps >= budget_steps:
                fitnesses = fitnesses[:i + 1]
                pop = pop[:i + 1]
                break

        info = gen_best_info or {"laps": 0}
        record(float(np.max(fitnesses)), {"laps": info.get("laps", 0), "pop": len(pop)})

        if env.total_steps >= budget_steps:
            break

        # Build the next generation: keep elites, fill the rest with offspring.
        order = np.argsort(fitnesses)[::-1]
        new_pop = [pop[order[k]].copy() for k in range(min(ELITE, len(pop)))]
        while len(new_pop) < POP:
            pa = pop[_tournament(rng, fitnesses)]
            pb = pop[_tournament(rng, fitnesses)]
            child = _mutate(rng, _crossover(rng, pa, pb))
            new_pop.append(child.astype(np.float32))
        pop = np.asarray(new_pop, dtype=np.float32)

    return best_fitness, best_params
