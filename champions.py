"""Persist and reload the best policy each technique finds.

Every technique searches a different kind of object -- NEAT a graph genome, GA/ES
a flat weight vector, DQN a torch network -- so each gets its own (de)serializer
and its own `act(obs) -> action` reconstructor. Everything else (watch.py's
replay, benchmark.py's saving) goes through this one module so the on-disk naming
stays consistent:

    champions/<track>-<algo>.<ext>     ext: neat=pkl, ga/es=npy, dqn=pt

This is what lets you train once (benchmark.py or watch.py) and re-watch the
driver later (watch.py --replay) without retraining.
"""

import os
import pickle

import numpy as np

import play
from techniques.mlp import MLPPolicy

ALGOS = ("NEAT", "GA", "ES", "DQN")
_EXT = {"neat": "pkl", "ga": "npy", "es": "npy", "dqn": "pt"}


def path(track, algo):
    algo = algo.lower()
    return os.path.join("champions", "%s-%s.%s" % (track, algo, _EXT[algo]))


def save(track, algo, params):
    """Persist a trained policy. `params` is whatever that technique's train()
    returns as its second value (neat genome / numpy vector / torch state_dict)."""
    algo = algo.lower()
    os.makedirs("champions", exist_ok=True)
    p = path(track, algo)
    if algo == "neat":
        with open(p, "wb") as f:
            pickle.dump(params, f)
    elif algo in ("ga", "es"):
        np.save(p, np.asarray(params, dtype=np.float32))
    elif algo == "dqn":
        import torch
        torch.save(params, p)
    else:
        raise ValueError("unknown algo %r" % algo)
    return p


def load_act(track, algo):
    """Rebuild the `act(obs) -> action int` callable for a saved champion, or
    return None if no champion has been saved for this (track, algo) yet."""
    algo = algo.lower()
    p = path(track, algo)
    if not os.path.exists(p):
        return None

    if algo == "neat":
        import neat
        with open(p, "rb") as f:
            genome = pickle.load(f)
        net = neat.nn.FeedForwardNetwork.create(genome, play.build_config())
        return lambda obs: int(np.argmax(net.activate(obs)))

    if algo in ("ga", "es"):
        policy = MLPPolicy(np.load(p), play.NUM_INPUTS, play.NUM_OUTPUTS)
        return policy.act

    if algo == "dqn":
        import torch
        from techniques.dqn import QNet
        q = QNet(play.NUM_INPUTS, play.NUM_OUTPUTS)
        q.load_state_dict(torch.load(p))
        q.eval()

        def act(obs):
            with torch.no_grad():
                t = torch.as_tensor(np.asarray(obs, dtype=np.float32)).unsqueeze(0)
                return int(q(t).argmax(dim=1).item())
        return act

    raise ValueError("unknown algo %r" % algo)
