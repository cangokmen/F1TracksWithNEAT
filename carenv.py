"""Shared headless environment that wraps play.py's Car as a reusable
reset()/step(action) interface.

Every learning technique in techniques/ drives the *same* environment so the
benchmark is apples-to-apples: identical physics, identical 8-dim observation
(7 radars + speed), identical 4 discrete actions, identical gate/distance
reward. The only thing that differs between techniques is how the policy
(obs -> action) is produced.

The environment is deliberately Gym-like but intentionally tiny -- no external
RL framework dependency. It counts every env step in `total_steps`, which the
benchmark uses as the fair cross-paradigm x-axis (a NEAT "generation" and a DQN
"episode" are not the same unit of work, but an env step is).
"""

import os

# Must be set before pygame is imported (play.py imports pygame at module load).
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import numpy as np
import pygame

import play

# Re-export the action count so techniques don't reach into play.py themselves.
NUM_INPUTS = play.NUM_INPUTS
NUM_OUTPUTS = play.NUM_OUTPUTS
TICK_LIMIT = play.GENERATION_TICK_LIMIT


class CarEnv:
    """One car driving one track, stepped at the same cadence as NEAT's loop.

    Reuses play.Car for physics/sensors/reward so this stays in lockstep with
    the live simulation. A single env instance can be reused across episodes
    via reset(); creating one loads the track image (expensive), so the
    benchmark makes one per algorithm and resets it between rollouts.
    """

    def __init__(self, track, max_ticks=TICK_LIMIT):
        if track not in play.track_dict:
            raise KeyError("Unknown track '%s'. Known: %s" % (track, sorted(play.track_dict)))
        self.track = track
        self.max_ticks = max_ticks

        pygame.init()
        # A display mode must exist for pygame's .convert() to work, even under
        # the dummy SDL driver (no real window opens).
        pygame.display.set_mode((play.WIDTH, play.HEIGHT))
        self.game_map = pygame.image.load(play.map_image_path(track)).convert()
        self.gates = play.load_gates(track)

        self.car = None
        self.ticks = 0
        self.total_steps = 0  # cumulative across all episodes -- the benchmark x-axis

    def reset(self):
        """Start a fresh episode; return the initial observation (all zeros,
        matching what a freshly spawned car sees before its first move -- the
        same first-tick observation NEAT's loop feeds the network)."""
        self.car = play.Car(self.track, self.gates)
        self.ticks = 0
        return np.asarray(self.car.get_data(), dtype=np.float32)

    def step(self, action):
        """Apply a discrete action (0=left,1=right,2=brake,3=accel), advance one
        tick, and return (obs, reward, done, info). Mirrors NEAT's per-tick order:
        decide -> apply_action -> update -> read new obs/reward."""
        play.apply_action(self.car, int(action))
        self.car.update(self.game_map)
        self.ticks += 1
        self.total_steps += 1

        obs = np.asarray(self.car.get_data(), dtype=np.float32)
        reward = self.car.last_reward
        done = (not self.car.is_alive()) or self.ticks >= self.max_ticks
        info = {
            "laps": self.car.laps,
            "gate": self.car.current_gate,
            "distance": self.car.distance,
            "alive": self.car.is_alive(),
        }
        return obs, reward, done, info


def rollout(env, act, max_steps=None):
    """Run one full episode under policy `act` (obs -> action int). Returns
    (total_reward, info_of_last_step). Used by GA/ES to score a policy and by
    every technique's logger to evaluate "best so far"."""
    obs = env.reset()
    total = 0.0
    done = False
    info = {"laps": 0, "gate": 0, "distance": 0.0, "alive": True}
    steps = 0
    while not done:
        obs, r, done, info = env.step(act(obs))
        total += r
        steps += 1
        if max_steps is not None and steps >= max_steps:
            break
    return total, info
