"""Benchmark NEAT vs GA vs Evolution Strategies vs DQN on the same track.

All four techniques drive the identical CarEnv (same physics, observation,
action set, and reward), so the only variable is the learning algorithm. Each is
given the same budget of environment steps -- the fair cross-paradigm unit of
work, since a NEAT "generation", a GA "generation", and a DQN "episode" cost
wildly different numbers of steps.

Outputs (into results/<track>/):
  - metrics.csv                     raw per-generation/episode log
  - fitness_vs_steps.png            sample efficiency (the headline chart)
  - fitness_vs_walltime.png         practical training speed
  - final_performance.png           best fitness reached, per technique

Usage:
  python3 benchmark.py silverstone                       # all four, default budget
  python3 benchmark.py monza --steps 400000 --seeds 3
  python3 benchmark.py spa --algos NEAT GA --quick       # fast smoke test
"""

import argparse
import csv
import os
import time
import warnings

import matplotlib
matplotlib.use("Agg")  # headless: render to files, never open a window
import matplotlib.pyplot as plt
import numpy as np

import champions
from carenv import CarEnv
from techniques import dqn, es, ga, neat_runner

TRAINERS = {
    "NEAT": neat_runner.train,
    "GA": ga.train,
    "ES": es.train,
    "DQN": dqn.train,
}

COLORS = {"NEAT": "#1f77b4", "GA": "#2ca02c", "ES": "#ff7f0e", "DQN": "#d62728"}

GRID_POINTS = 200


def run_one(name, track, budget, seed):
    """Train one technique once; return its per-iteration log as a list of rows."""
    env = CarEnv(track)
    rows = []
    t0 = time.time()

    def record(fitness, info):
        rows.append({
            "env_steps": env.total_steps,
            "wall": time.time() - t0,
            "fitness": float(fitness),
            "laps": info.get("laps", 0),
        })

    best_fitness, best_params = TRAINERS[name](env, budget, seed, record)
    print("  %-5s seed %d: best fitness=%.1f  (%d iters, %.1fs, %d steps)"
          % (name, seed, best_fitness, len(rows), time.time() - t0, env.total_steps))
    return rows, best_fitness, best_params


def best_so_far_on_grid(rows, xkey, grid):
    """Step-interpolate a run's running-best fitness onto a shared x grid.
    Points before the run's first evaluation are NaN (so they don't drag means)."""
    xs = np.array([r[xkey] for r in rows], dtype=np.float64)
    ys = np.maximum.accumulate(np.array([r["fitness"] for r in rows], dtype=np.float64))
    idx = np.searchsorted(xs, grid, side="right") - 1
    out = np.full(len(grid), np.nan)
    valid = idx >= 0
    out[valid] = ys[idx[valid]]
    return out


def aggregate(runs, xkey, grid):
    """Mean and min/max band of running-best fitness across seeds, on `grid`."""
    curves = np.vstack([best_so_far_on_grid(r, xkey, grid) for r in runs])
    # Early grid points can be all-NaN (before any run's first eval); that's
    # expected, so silence the empty-slice warnings rather than spam stderr.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        mean = np.nanmean(curves, axis=0)
        lo = np.nanmin(curves, axis=0)
        hi = np.nanmax(curves, axis=0)
    return mean, lo, hi


def plot_curves(results, xkey, xlabel, title, path, budget):
    grid_max = budget if xkey == "env_steps" else max(
        r["wall"] for runs in results.values() for run in runs for r in run)
    grid = np.linspace(0, grid_max, GRID_POINTS)

    plt.figure(figsize=(9, 6))
    for name, runs in results.items():
        mean, lo, hi = aggregate(runs, xkey, grid)
        plt.plot(grid, mean, label=name, color=COLORS[name], linewidth=2)
        if len(runs) > 1:
            plt.fill_between(grid, lo, hi, color=COLORS[name], alpha=0.15)
    plt.xlabel(xlabel)
    plt.ylabel("Best fitness so far (episode return)")
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=130)
    plt.close()
    print("  wrote", path)


def plot_final(results, track, path):
    names = list(results.keys())
    finals = {n: [np.maximum.accumulate([r["fitness"] for r in run])[-1]
                  for run in runs] for n, runs in results.items()}
    means = [np.mean(finals[n]) for n in names]
    errs = [(np.mean(finals[n]) - np.min(finals[n]),
             np.max(finals[n]) - np.mean(finals[n])) for n in names]
    yerr = np.array(errs).T if any(len(finals[n]) > 1 for n in names) else None

    plt.figure(figsize=(8, 6))
    plt.bar(names, means, yerr=yerr, capsize=6,
            color=[COLORS[n] for n in names])
    for i, n in enumerate(names):
        plt.text(i, means[i], "%.0f" % means[i], ha="center", va="bottom")
    plt.ylabel("Best fitness reached")
    plt.title("Final performance on %s (mean over seeds; bars = min/max)" % track)
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=130)
    plt.close()
    print("  wrote", path)


def write_csv(results, path):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["algo", "seed_index", "iteration", "env_steps", "wall_seconds", "fitness", "laps"])
        for name, runs in results.items():
            for si, run in enumerate(runs):
                for it, row in enumerate(run):
                    w.writerow([name, si, it, row["env_steps"],
                                "%.3f" % row["wall"], "%.3f" % row["fitness"], row["laps"]])
    print("  wrote", path)


def main():
    p = argparse.ArgumentParser(description="Benchmark AI techniques on an F1 track.")
    p.add_argument("track")
    p.add_argument("--algos", nargs="+", default=list(TRAINERS),
                   choices=list(TRAINERS), help="techniques to run")
    p.add_argument("--steps", type=int, default=300_000, help="env-step budget per technique per seed")
    p.add_argument("--seeds", type=int, default=1, help="independent runs per technique")
    p.add_argument("--quick", action="store_true", help="tiny budget smoke test (20k steps)")
    p.add_argument("--out", default="results", help="output directory root")
    args = p.parse_args()

    budget = 20_000 if args.quick else args.steps
    out_dir = os.path.join(args.out, args.track)
    os.makedirs(out_dir, exist_ok=True)

    print("Benchmark on %s | budget=%d steps/seed | seeds=%d | algos=%s"
          % (args.track, budget, args.seeds, ", ".join(args.algos)))

    results = {}
    for name in args.algos:
        print(name + ":")
        runs, champ_fit, champ_params = [], -np.inf, None
        for seed in range(args.seeds):
            rows, best_fit, best_params = run_one(name, args.track, budget, seed)
            runs.append(rows)
            if best_fit > champ_fit and best_params is not None:
                champ_fit, champ_params = best_fit, best_params
        results[name] = runs
        if champ_params is not None:
            print("  saved %s champion (fitness=%.1f) to %s"
                  % (name, champ_fit, champions.save(args.track, name, champ_params)))

    write_csv(results, os.path.join(out_dir, "metrics.csv"))
    plot_curves(results, "env_steps", "Environment steps",
                "Sample efficiency on %s" % args.track,
                os.path.join(out_dir, "fitness_vs_steps.png"), budget)
    plot_curves(results, "wall", "Wall-clock seconds",
                "Training speed on %s" % args.track,
                os.path.join(out_dir, "fitness_vs_walltime.png"), budget)
    plot_final(results, args.track, os.path.join(out_dir, "final_performance.png"))
    print("Done. See", out_dir)


if __name__ == "__main__":
    main()
