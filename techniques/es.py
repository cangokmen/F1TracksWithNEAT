"""Evolution Strategies (OpenAI-ES style) over the same fixed MLP.

Where the GA keeps a population and recombines survivors, ES maintains a single
parameter vector theta and estimates a search gradient by probing many small
random perturbations of it (mirrored / antithetic sampling), then takes a step
in the fitness-weighted direction. No backprop, no replay -- but unlike the GA
it follows a smoothed gradient, which often climbs smooth landscapes faster.

Uses centered-rank fitness shaping (the standard OpenAI-ES trick) so a single
lucky high-reward episode can't dominate the update.

Trainer contract matches techniques/ga.py:
    train(env, budget_steps, seed, record) -> (best_fitness, best_params)
"""

import numpy as np

from carenv import NUM_INPUTS, NUM_OUTPUTS, rollout
from techniques.mlp import MLPPolicy, param_count

N_DIRECTIONS = 20    # perturbation directions per iteration (2x evals via mirroring)
SIGMA = 0.15         # exploration noise std
LR = 0.05            # step size


def _evaluate(env, theta):
    policy = MLPPolicy(theta, NUM_INPUTS, NUM_OUTPUTS)
    return rollout(env, policy.act)


def _centered_ranks(x):
    """Map values to centered ranks in [-0.5, 0.5]; robust to reward scale."""
    ranks = np.empty(len(x), dtype=np.float64)
    ranks[np.argsort(x)] = np.arange(len(x))
    return ranks / (len(x) - 1) - 0.5


def train(env, budget_steps, seed, record):
    rng = np.random.default_rng(seed)
    dim = param_count(NUM_INPUTS, 12, NUM_OUTPUTS)
    theta = rng.normal(0.0, 1.0, size=dim).astype(np.float32)

    best_fitness = -np.inf
    best_params = theta.copy()

    while env.total_steps < budget_steps:
        noise = rng.normal(0.0, 1.0, size=(N_DIRECTIONS, dim)).astype(np.float32)
        full = np.concatenate([noise, -noise], axis=0)  # antithetic pairs

        evals = np.empty(len(full), dtype=np.float64)
        gen_best_info = None
        for i, eps in enumerate(full):
            evals[i], info = _evaluate(env, theta + SIGMA * eps)
            if evals[i] > best_fitness:
                best_fitness = evals[i]
                best_params = (theta + SIGMA * eps).copy()
                gen_best_info = info
            if env.total_steps >= budget_steps:
                evals = evals[:i + 1]
                full = full[:i + 1]
                break

        # Fitness-shaped search-gradient ascent step.
        utilities = _centered_ranks(evals)
        grad = (utilities @ full) / (len(full) * SIGMA)
        theta = (theta + LR * grad).astype(np.float32)

        info = gen_best_info or {"laps": 0}
        record(float(np.max(evals)), {"laps": info.get("laps", 0)})

    return best_fitness, best_params
