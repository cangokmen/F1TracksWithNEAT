"""Deep Q-Network (PyTorch).

The odd one out: a gradient-based reinforcement learner, not an evolutionary
method. Instead of scoring whole episodes and selecting, DQN learns a value
Q(obs, action) from *individual transitions* via temporal-difference bootstrapping,
explores with epsilon-greedy, and stabilizes training with a replay buffer and a
periodically-synced target network. This is the most different paradigm in the
benchmark and the most sample-efficient when it works.

Trainer contract matches the evolutionary trainers:
    train(env, budget_steps, seed, record) -> (best_fitness, best_state_dict)
record() is called once per finished episode with that episode's return (sum of
the *same* env reward the other methods maximize, so the charts are comparable).
"""

import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn

from carenv import NUM_INPUTS, NUM_OUTPUTS

HIDDEN = 128
GAMMA = 0.99
LR = 1e-3
BATCH = 64
BUFFER_CAP = 50_000
WARMUP = 1_000          # steps of random play before learning starts
TARGET_SYNC = 1_000     # hard target-net update interval (steps)
EPS_START, EPS_END = 1.0, 0.05
EPS_FRACTION = 0.4      # fraction of budget over which epsilon decays


class QNet(nn.Module):
    def __init__(self, n_in, n_out, hidden=HIDDEN):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_in, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, n_out),
        )

    def forward(self, x):
        return self.net(x)


def train(env, budget_steps, seed, record):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device("cpu")

    q = QNet(NUM_INPUTS, NUM_OUTPUTS).to(device)
    target = QNet(NUM_INPUTS, NUM_OUTPUTS).to(device)
    target.load_state_dict(q.state_dict())
    opt = torch.optim.Adam(q.parameters(), lr=LR)
    buffer = deque(maxlen=BUFFER_CAP)

    eps_decay_steps = max(1, int(budget_steps * EPS_FRACTION))

    def epsilon():
        frac = min(1.0, env.total_steps / eps_decay_steps)
        return EPS_START + frac * (EPS_END - EPS_START)

    def select(obs):
        if random.random() < epsilon():
            return random.randrange(NUM_OUTPUTS)
        with torch.no_grad():
            t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            return int(q(t).argmax(dim=1).item())

    def learn():
        if len(buffer) < max(WARMUP, BATCH):
            return
        batch = random.sample(buffer, BATCH)
        obs, act, rew, nxt, done = zip(*batch)
        obs = torch.as_tensor(np.array(obs), dtype=torch.float32, device=device)
        act = torch.as_tensor(act, dtype=torch.int64, device=device).unsqueeze(1)
        rew = torch.as_tensor(rew, dtype=torch.float32, device=device).unsqueeze(1)
        nxt = torch.as_tensor(np.array(nxt), dtype=torch.float32, device=device)
        done = torch.as_tensor(done, dtype=torch.float32, device=device).unsqueeze(1)

        qsa = q(obs).gather(1, act)
        with torch.no_grad():
            target_max = target(nxt).max(dim=1, keepdim=True).values
            y = rew + GAMMA * target_max * (1.0 - done)
        loss = nn.functional.smooth_l1_loss(qsa, y)
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(q.parameters(), 10.0)
        opt.step()

    best_fitness = -np.inf
    best_state = {k: v.clone() for k, v in q.state_dict().items()}
    last_sync = 0

    while env.total_steps < budget_steps:
        obs = env.reset()
        done = False
        ep_ret = 0.0
        info = {"laps": 0}
        while not done:
            a = select(obs)
            nxt, r, done, info = env.step(a)
            buffer.append((obs, a, r, nxt, float(done)))
            obs = nxt
            ep_ret += r

            learn()
            if env.total_steps - last_sync >= TARGET_SYNC:
                target.load_state_dict(q.state_dict())
                last_sync = env.total_steps
            if env.total_steps >= budget_steps:
                break

        if ep_ret > best_fitness:
            best_fitness = ep_ret
            best_state = {k: v.clone() for k, v in q.state_dict().items()}
        record(ep_ret, {"laps": info.get("laps", 0), "epsilon": round(epsilon(), 3)})

    return best_fitness, best_state
