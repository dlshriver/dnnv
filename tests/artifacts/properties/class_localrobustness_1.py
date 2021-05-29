from dnnv.properties import *
import numpy as np
import os

INPUT_LAYER = eval(os.environ.get("INPUT_LAYER", "None"))
OUTPUT_LAYER = eval(os.environ.get("OUTPUT_LAYER", "None"))
N = Network()[INPUT_LAYER:OUTPUT_LAYER]

x = np.random.randn(1, N.input_shape[0][1]).astype(np.float32)
x = x + eval(os.environ.get("SHIFT", "0"))
epsilon = 3.0

Forall(
    x_, Implies(x - epsilon <= x_ <= x + epsilon, np.argmax(N(x_)) == np.argmax(N(x)))
)
# label = np.argmax(N(x))
# other_label = 1 - label
# Forall(
#     x_,
#     Implies(
#         (x - epsilon) <= x_ <= (x + epsilon), N(x_)[0][label] >= N(x_)[0][other_label]
#     ),
# )
