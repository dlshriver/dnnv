import ast
import os

import numpy as np

from dnnv.properties import *

rng = np.random.default_rng(int(os.environ.get("SEED", "0")))

INPUT_LAYER = eval(os.environ.get("INPUT_LAYER", "None"))
OUTPUT_LAYER = eval(os.environ.get("OUTPUT_LAYER", "None"))
N = Network("N")[INPUT_LAYER:OUTPUT_LAYER]

x = rng.normal(size=N.input_shape[0][1])[None].astype(np.float32)
x = x + np.asarray(ast.literal_eval(os.environ.get("SHIFT", "0"))).astype(np.float32)
y = np.argmax(N(x))
epsilon = 3.0

Forall(
    x_,
    Implies(
        x - epsilon <= x_ <= x + epsilon,
        And(
            Implies(y != 0, N(x_)[0, 0] < N(x_)[0][y]),
            Implies(y != 1, N(x_)[0, 1] < N(x_)[0][y]),
        ),
    ),
)
