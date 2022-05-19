import os

import numpy as np

from dnnv.properties import *

rng = np.random.default_rng(int(os.environ.get("SEED", "0")))

INPUT_LAYER = eval(os.environ.get("INPUT_LAYER", "None"))
OUTPUT_LAYER = eval(os.environ.get("OUTPUT_LAYER", "None"))
N = Network("N")[INPUT_LAYER:OUTPUT_LAYER]

x = rng.normal(
    loc=eval(os.environ.get("SHIFT", "0")),
    scale=eval(os.environ.get("SCALE", "1")),
    size=(1,) + N.input_shape[0][1:],
).astype(np.float32)
epsilon = eval(os.environ.get("EPSILON", "0.01"))

Forall(x_, Implies(x - epsilon <= x_ <= x + epsilon, N(x_)[0, 0] + 1.0 > np.max(x)))
