import os

import numpy as np

from dnnv.properties import *

rng = np.random.default_rng(int(os.environ.get("SEED", "0")))

INPUT_LAYER = eval(os.environ.get("INPUT_LAYER", "None"))
OUTPUT_LAYER = eval(os.environ.get("OUTPUT_LAYER", "None"))
N = Network("N")[INPUT_LAYER:OUTPUT_LAYER]

x = rng.normal(size=(1, 1, 28, 28)).astype(np.float32)
epsilon = float(os.environ.get("EPSILON", "0.007843137254902"))

Forall(
    x_, Implies(x - epsilon <= x_ <= x + epsilon, np.argmax(N(x_)) == np.argmax(N(x)))
)
