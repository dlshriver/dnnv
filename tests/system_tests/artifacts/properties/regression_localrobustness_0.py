import ast
import os

import numpy as np

from dnnv.properties import *

rng = np.random.default_rng(int(os.environ.get("SEED", "0")))

INPUT_LAYER = ast.literal_eval(os.environ.get("INPUT_LAYER", "None"))
OUTPUT_LAYER = ast.literal_eval(os.environ.get("OUTPUT_LAYER", "None"))
N = Network("N")[INPUT_LAYER:OUTPUT_LAYER]

x = rng.normal(size=N.input_shape[0][1])[None].astype(np.float32)
x = x + eval(os.environ.get("SHIFT", "0"))
epsilon = 3.0

Forall(
    x_,
    Implies(
        x - epsilon <= x_ <= x + epsilon,
        And(Implies(N(x) > 0, N(x_) > 0), Implies(N(x) <= 0, N(x_) <= 0)),
    ),
)
