from dnnv.properties import *
import numpy as np
import os

seed = np.random.seed(int(os.environ.get("SEED", "0")))

INPUT_LAYER = eval(os.environ.get("INPUT_LAYER", "None"))
OUTPUT_LAYER = eval(os.environ.get("OUTPUT_LAYER", "None"))
N = Network()[INPUT_LAYER:OUTPUT_LAYER]

x = np.random.normal(
    loc=eval(os.environ.get("SHIFT", "0")),
    scale=eval(os.environ.get("SCALE", "0")),
    size=(1,) + N.input_shape[0][1:],
).astype(np.float32)
y = np.argmax(N(x))
epsilon = eval(os.environ.get("EPSILON", "0.01"))

Forall(
    x_,
    Implies(
        And(
            x - 2.0 <= x_ <= x + 2.0,
            x[0, 0, 0, 0] - epsilon <= x_[0, 0, 0, 0] <= x[0, 0, 0, 0] + epsilon,
            x[0, 0, 0, 2] - epsilon <= x_[0, 0, 0, 2] <= x[0, 0, 0, 2] + epsilon,
            x[0, 0, 1, 0] - epsilon <= x_[0, 0, 1, 0] <= x[0, 0, 1, 0] + epsilon,
            x[0, 0, 1, 1] - epsilon <= x_[0, 0, 1, 1] <= x[0, 0, 1, 1] + epsilon,
            x[0, 0, 2, 1] - epsilon <= x_[0, 0, 2, 1] <= x[0, 0, 2, 1] + epsilon,
            x[0, 0, 2, 2] - epsilon <= x_[0, 0, 2, 2] <= x[0, 0, 2, 2] + epsilon,
        ),
        np.argmax(N(x_)) == np.argmax(N(x)),
    ),
)
