"""Watch any technique learn to drive -- live, in a window.

    python3 watch.py <track> --algo {neat,ga,es,dqn}            # watch it train
    python3 watch.py <track> --algo ga --replay                 # watch the saved champion

This is the visual counterpart to the headless benchmark.py. Same physics, same
observation, same reward, same algorithms (it reuses the evolution operators in
techniques/ and dqn.train itself) -- the only thing added is a real pygame window
so you can see what the search is doing.

How each paradigm is shown while training:
  - NEAT / GA / ES are population methods: every candidate in the current
    generation drives the track *simultaneously* (like play.py does for NEAT),
    so you watch the whole swarm get better generation by generation.
  - DQN is a single agent, so you watch that one car drive episode after episode
    as epsilon decays and the replay buffer fills.

Whenever a new best policy is found it is saved via champions.py, so you can
re-watch it later with --replay (and benchmark.py writes the same files).

Keys:  SPACE pause/resume   F toggle fast (uncapped, skips drawing)   ESC/close quit
"""

import argparse
import os
import sys

# watch.py ALWAYS opens a real window. carenv.py does
# os.environ.setdefault("SDL_VIDEODRIVER", "dummy") at import, which would route
# every later set_mode to the headless dummy driver and no window would appear.
# Claim a real driver here, before any import that pulls in carenv/play. Only do
# so if nothing is set yet (in normal use it is genuinely unset at this point,
# since carenv has not been imported); an explicit SDL_VIDEODRIVER is respected,
# which also lets `SDL_VIDEODRIVER=dummy python3 watch.py ...` run headless for tests.
if "SDL_VIDEODRIVER" not in os.environ:
    if sys.platform == "darwin":
        os.environ["SDL_VIDEODRIVER"] = "cocoa"
    elif sys.platform.startswith("win"):
        os.environ["SDL_VIDEODRIVER"] = "windows"
    else:
        os.environ["SDL_VIDEODRIVER"] = "x11"
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import numpy as np
import pygame

import champions
import play
from techniques import ga as ga_mod
from techniques import es as es_mod
from techniques.mlp import HIDDEN, MLPPolicy, param_count

NUM_INPUTS = play.NUM_INPUTS
NUM_OUTPUTS = play.NUM_OUTPUTS
TICK_LIMIT = play.GENERATION_TICK_LIMIT


class Quit(Exception):
    """Raised from the renderer when the user closes the window / hits ESC, to
    unwind out of whatever training loop is currently running."""


class Renderer:
    """Owns the one real window. Draws the track once per frame with every alive
    car on top, scaled to fit the screen so the full 1800x1200 map is visible."""

    def __init__(self, track, scale_to=1280, fps=60):
        self.scale = min(1.0, scale_to / play.WIDTH)
        self.win_size = (int(play.WIDTH * self.scale), int(play.HEIGHT * self.scale))
        pygame.init()
        pygame.font.init()
        self.screen = pygame.display.set_mode(self.win_size)
        pygame.display.set_caption("watch.py - %s" % track)
        self.canvas = pygame.Surface((play.WIDTH, play.HEIGHT))
        self.track = track
        self.game_map = pygame.image.load(play.map_image_path(track)).convert()
        self.gates = play.load_gates(track)
        self.font = pygame.font.SysFont("Chicago", 30)
        self.clock = pygame.time.Clock()
        self.fps = fps
        self.paused = False
        self.fast = False

    def _pump(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                raise Quit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    raise Quit()
                if event.key == pygame.K_SPACE:
                    self.paused = not self.paused
                if event.key == pygame.K_f:
                    self.fast = not self.fast

    def draw(self, cars, hud_lines):
        """Pump events, then (unless in fast mode) paint one frame. Blocks here
        while paused so the sim freezes but the window stays responsive."""
        self._pump()
        while self.paused:
            self._pump()
            self.clock.tick(15)

        if self.fast:
            return

        self.canvas.blit(self.game_map, (0, 0))
        for gate in self.gates:
            pygame.draw.circle(self.canvas, (0, 180, 0), (gate["x"], gate["y"]), gate["r"], 2)
        for car in cars:
            if car.is_alive():
                car.draw(self.canvas)
        for i, line in enumerate(hud_lines):
            self.canvas.blit(self.font.render(line, True, (0, 0, 0)), (20, 20 + i * 34))

        pygame.transform.scale(self.canvas, self.win_size, self.screen)
        pygame.display.flip()
        self.clock.tick(self.fps)


def evaluate_generation(renderer, acts, hud_prefix, max_ticks):
    """Drive one car per policy in `acts` simultaneously until all crash or
    max_ticks elapses, rendering every frame. Returns each car's total reward
    and last info -- identical accounting to carenv.rollout, just run in lockstep
    so the whole generation can be drawn at once.

    `hud_prefix` is a list of HUD lines describing the run (algo, generation,
    best-so-far); per-frame status (alive count, leader) is appended."""
    cars = [play.Car(renderer.track, renderer.gates) for _ in acts]
    totals = [0.0] * len(acts)
    infos = [{"laps": 0, "gate": 0, "distance": 0.0} for _ in acts]

    for tick in range(max_ticks):
        for i, car in enumerate(cars):
            if not car.is_alive():
                continue
            obs = np.asarray(car.get_data(), dtype=np.float32)
            play.apply_action(car, int(acts[i](obs)))

        alive = 0
        for i, car in enumerate(cars):
            if car.is_alive():
                alive += 1
                car.update(renderer.game_map)
                totals[i] += car.last_reward
                infos[i] = {"laps": car.laps, "gate": car.current_gate, "distance": car.distance}

        if alive == 0:
            break

        leader = int(np.argmax(totals))
        hud = hud_prefix + [
            "tick %d/%d   alive %d/%d" % (tick + 1, max_ticks, alive, len(cars)),
            "leader: fit %.0f  laps %d  gate %d" % (totals[leader], infos[leader]["laps"], infos[leader]["gate"]),
        ]
        renderer.draw(cars, hud)

    return totals, infos


def make_net_acts(genomes, config):
    import neat
    acts = []
    for _, g in genomes:
        net = neat.nn.FeedForwardNetwork.create(g, config)
        acts.append(lambda obs, net=net: int(np.argmax(net.activate(obs))))
    return acts


def watch_neat(renderer, track, gens, max_ticks):
    import neat
    config = play.build_config()
    config.no_fitness_termination = True
    pop = neat.Population(config)
    best = {"fitness": -np.inf, "genome": None}

    def eval_genomes(genomes, cfg):
        acts = make_net_acts(genomes, cfg)
        totals, _ = evaluate_generation(
            renderer, acts,
            ["NEAT  gen %d" % (renderer.gen + 1), "best so far: %.0f" % best["fitness"]],
            max_ticks)
        for (gid, g), fit in zip(genomes, totals):
            g.fitness = fit
            if fit > best["fitness"]:
                best["fitness"] = fit
                best["genome"] = g
        renderer.gen += 1

    renderer.gen = 0
    for _ in range(gens):
        pop.run(eval_genomes, 1)
        if best["genome"] is not None:
            champions.save(track, "NEAT", best["genome"])
    print("NEAT done. best fitness=%.1f saved to %s" % (best["fitness"], champions.path(track, "NEAT")))


def watch_ga(renderer, track, gens, max_ticks, seed):
    rng = np.random.default_rng(seed)
    dim = param_count(NUM_INPUTS, HIDDEN, NUM_OUTPUTS)
    pop = rng.normal(0.0, ga_mod.INIT_STD, size=(ga_mod.POP, dim)).astype(np.float32)
    best_fitness, best_params = -np.inf, pop[0].copy()

    for gen in range(gens):
        acts = [MLPPolicy(ind, NUM_INPUTS, NUM_OUTPUTS).act for ind in pop]
        totals, _ = evaluate_generation(
            renderer, acts,
            ["GA  gen %d/%d" % (gen + 1, gens), "best so far: %.0f" % best_fitness], max_ticks)
        fitnesses = np.asarray(totals, dtype=np.float64)
        gi = int(np.argmax(fitnesses))
        if fitnesses[gi] > best_fitness:
            best_fitness, best_params = float(fitnesses[gi]), pop[gi].copy()
            champions.save(track, "GA", best_params)

        # Reproduce: elitism + tournament-selected, crossed-over, mutated offspring
        # (identical operators to techniques/ga.py).
        order = np.argsort(fitnesses)[::-1]
        new_pop = [pop[order[k]].copy() for k in range(min(ga_mod.ELITE, len(pop)))]
        while len(new_pop) < ga_mod.POP:
            pa = pop[ga_mod._tournament(rng, fitnesses)]
            pb = pop[ga_mod._tournament(rng, fitnesses)]
            child = ga_mod._mutate(rng, ga_mod._crossover(rng, pa, pb))
            new_pop.append(child.astype(np.float32))
        pop = np.asarray(new_pop, dtype=np.float32)

    print("GA done. best fitness=%.1f saved to %s" % (best_fitness, champions.path(track, "GA")))


def watch_es(renderer, track, gens, max_ticks, seed):
    rng = np.random.default_rng(seed)
    dim = param_count(NUM_INPUTS, HIDDEN, NUM_OUTPUTS)
    theta = rng.normal(0.0, 1.0, size=dim).astype(np.float32)
    best_fitness, best_params = -np.inf, theta.copy()

    for gen in range(gens):
        noise = rng.normal(0.0, 1.0, size=(es_mod.N_DIRECTIONS, dim)).astype(np.float32)
        full = np.concatenate([noise, -noise], axis=0)  # antithetic pairs
        acts = [MLPPolicy(theta + es_mod.SIGMA * eps, NUM_INPUTS, NUM_OUTPUTS).act for eps in full]
        totals, _ = evaluate_generation(
            renderer, acts,
            ["ES  gen %d/%d" % (gen + 1, gens), "best so far: %.0f" % best_fitness], max_ticks)
        evals = np.asarray(totals, dtype=np.float64)
        bi = int(np.argmax(evals))
        if evals[bi] > best_fitness:
            best_fitness, best_params = float(evals[bi]), (theta + es_mod.SIGMA * full[bi]).copy()
            champions.save(track, "ES", best_params)

        # Fitness-shaped search-gradient ascent step (identical to techniques/es.py).
        utilities = es_mod._centered_ranks(evals)
        grad = (utilities @ full) / (len(full) * es_mod.SIGMA)
        theta = (theta + es_mod.LR * grad).astype(np.float32)

    print("ES done. best fitness=%.1f saved to %s" % (best_fitness, champions.path(track, "ES")))


class RenderingCarEnv:
    """A drop-in for carenv.CarEnv that renders the (single) car each step, so
    dqn.train() runs visibly with no changes. DQN learns mid-episode and has no
    population, so the natural view is just this one agent driving."""

    def __init__(self, renderer, track, max_ticks):
        self.renderer = renderer
        self.track = track
        self.max_ticks = max_ticks
        self.game_map = renderer.game_map
        self.gates = renderer.gates
        self.car = None
        self.ticks = 0
        self.total_steps = 0
        self.episode = 0
        self.best = -np.inf

    def reset(self):
        self.car = play.Car(self.track, self.gates)
        self.ticks = 0
        self.episode += 1
        self._ep_ret = 0.0
        return np.asarray(self.car.get_data(), dtype=np.float32)

    def step(self, action):
        play.apply_action(self.car, int(action))
        self.car.update(self.game_map)
        self.ticks += 1
        self.total_steps += 1
        self._ep_ret += self.car.last_reward

        obs = np.asarray(self.car.get_data(), dtype=np.float32)
        done = (not self.car.is_alive()) or self.ticks >= self.max_ticks
        info = {"laps": self.car.laps, "gate": self.car.current_gate,
                "distance": self.car.distance, "alive": self.car.is_alive()}
        self.best = max(self.best, self._ep_ret)
        self.renderer.draw([self.car], [
            "DQN  episode %d" % self.episode,
            "steps %d/%d   best ep %.0f" % (self.total_steps, self._budget, self.best),
            "this ep: fit %.0f  laps %d  gate %d" % (self._ep_ret, self.car.laps, self.car.current_gate),
        ])
        return obs, self.car.last_reward, done, info


def watch_dqn(renderer, track, steps, max_ticks, seed):
    from techniques import dqn
    env = RenderingCarEnv(renderer, track, max_ticks)
    env._budget = steps

    def record(fitness, info):
        pass  # progress is shown live in the HUD; nothing to log here

    best_fitness, best_state = dqn.train(env, steps, seed, record)
    champions.save(track, "DQN", best_state)
    print("DQN done. best fitness=%.1f saved to %s" % (best_fitness, champions.path(track, "DQN")))


def replay(renderer, track, algo, max_ticks, episodes=0):
    act = champions.load_act(track, algo)
    if act is None:
        sys.exit("No saved %s champion for '%s'. Train one first: python3 watch.py %s --algo %s"
                 % (algo, track, track, algo.lower()))
    print("Replaying %s champion on %s. Close the window or press ESC to stop." % (algo, track))
    episode = 0
    while episodes == 0 or episode < episodes:
        episode += 1
        totals, infos = evaluate_generation(
            renderer, [act], ["%s replay  episode %d" % (algo, episode)], max_ticks)
        print("  episode %d: fitness=%.0f laps=%d gate=%d distance=%d"
              % (episode, totals[0], infos[0]["laps"], infos[0]["gate"], int(infos[0]["distance"])))


def parse_args():
    p = argparse.ArgumentParser(description="Watch an AI technique learn to drive an F1 track.")
    p.add_argument("track", choices=sorted(play.track_dict.keys()))
    p.add_argument("--algo", required=True, choices=["neat", "ga", "es", "dqn"],
                   help="which technique to watch")
    p.add_argument("--replay", action="store_true",
                   help="load the saved champion for this (track, algo) and just watch it drive")
    p.add_argument("--episodes", type=int, default=0,
                   help="number of replay episodes before exiting (0 = loop until you close the window)")
    p.add_argument("--gens", type=int, default=30,
                   help="generations to train (NEAT/GA/ES; default 30)")
    p.add_argument("--steps", type=int, default=80_000,
                   help="env-step budget to train (DQN; default 80000)")
    p.add_argument("--max-ticks", type=int, default=TICK_LIMIT, dest="max_ticks",
                   help="max ticks per generation/episode before reset (default %d)" % TICK_LIMIT)
    p.add_argument("--fps", type=int, default=60, help="frame rate cap (default 60; press F to uncap)")
    p.add_argument("--scale", type=int, default=1280, help="max window width in pixels (default 1280)")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main():
    args = parse_args()
    renderer = Renderer(args.track, scale_to=args.scale, fps=args.fps)
    algo = args.algo.upper()
    try:
        if args.replay:
            replay(renderer, args.track, algo, args.max_ticks, args.episodes)
        elif algo == "NEAT":
            watch_neat(renderer, args.track, args.gens, args.max_ticks)
        elif algo == "GA":
            watch_ga(renderer, args.track, args.gens, args.max_ticks, args.seed)
        elif algo == "ES":
            watch_es(renderer, args.track, args.gens, args.max_ticks, args.seed)
        elif algo == "DQN":
            watch_dqn(renderer, args.track, args.steps, args.max_ticks, args.seed)
    except Quit:
        print("Closed.")
    finally:
        pygame.quit()


if __name__ == "__main__":
    main()
