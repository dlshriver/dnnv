from dnnv.properties import *
import numpy as np
import os

INPUT_LAYER = eval(os.environ.get("INPUT_LAYER", "None"))
OUTPUT_LAYER = eval(os.environ.get("OUTPUT_LAYER", "None"))
N = Network()[INPUT_LAYER:OUTPUT_LAYER]

seed = np.random.seed(int(os.environ.get("SEED", "0")))
x = np.random.randn(1, 1, 28, 28).astype(np.float32)
epsilon = float(os.environ.get("EPSILON", "0.01"))

Forall(
    x_, Implies(x - epsilon <= x_ <= x + epsilon, np.argmax(N(x_)) == np.argmax(N(x)))
)
