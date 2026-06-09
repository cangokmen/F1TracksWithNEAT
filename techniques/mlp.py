"""A tiny fixed-topology MLP policy in pure numpy.

Both the genetic algorithm (techniques/ga.py) and evolution strategies
(techniques/es.py) optimize the *weights* of this exact network. That is the
whole point of the comparison against NEAT: NEAT grows its own topology, while
GA and ES are handed a fixed shape and may only tune the numbers in it.

The genome is a flat 1-D vector so the search algorithms never need to know the
layer structure -- they perturb/recombine a vector, hand it to `act`, done.
"""

import numpy as np

# 8 inputs (7 radars + speed) -> hidden -> 4 action logits. One hidden layer of
# 12 units keeps the parameter vector small (~160 numbers) so evolutionary
# search stays tractable while still being expressive enough to corner.
HIDDEN = 12


def layer_shapes(n_in, n_hidden, n_out):
    # (W1, b1, W2, b2)
    return [(n_in, n_hidden), (n_hidden,), (n_hidden, n_out), (n_out,)]


def param_count(n_in, n_hidden, n_out):
    return sum(int(np.prod(s)) for s in layer_shapes(n_in, n_hidden, n_out))


class MLPPolicy:
    """Deterministic argmax policy: obs -> action index in [0, n_out).

    Construct with a flat weight vector of length param_count(...). `act(obs)`
    is what gets handed to carenv.rollout / env.step.
    """

    def __init__(self, weights, n_in, n_out, n_hidden=HIDDEN):
        self.n_in, self.n_hidden, self.n_out = n_in, n_hidden, n_out
        self.shapes = layer_shapes(n_in, n_hidden, n_out)
        self._unpack(np.asarray(weights, dtype=np.float32))

    def _unpack(self, flat):
        out, i = [], 0
        for shape in self.shapes:
            size = int(np.prod(shape))
            out.append(flat[i:i + size].reshape(shape))
            i += size
        self.W1, self.b1, self.W2, self.b2 = out

    def forward(self, obs):
        h = np.tanh(obs @ self.W1 + self.b1)
        return h @ self.W2 + self.b2  # raw logits

    def act(self, obs):
        return int(np.argmax(self.forward(obs)))
