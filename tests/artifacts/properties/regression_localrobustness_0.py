from dnnv.properties import *
import ast
import numpy as np
import os

INPUT_LAYER = ast.literal_eval(os.environ.get("INPUT_LAYER", "None"))
OUTPUT_LAYER = ast.literal_eval(os.environ.get("OUTPUT_LAYER", "None"))
N = Network()[INPUT_LAYER:OUTPUT_LAYER]

x = np.random.randn(1, N.input_shape[0][1]).astype(np.float32)
x = x + eval(os.environ.get("SHIFT", "0"))
epsilon = 3.0

Forall(
    x_,
    Implies(
        x - epsilon <= x_ <= x + epsilon,
        And(Implies(N(x) > 0, N(x_) > 0), Implies(N(x) <= 0, N(x_) <= 0)),
    ),
)
