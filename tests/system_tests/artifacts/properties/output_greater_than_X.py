import os

import numpy as np

from dnnv.properties import *

INPUT_LAYER = eval(os.environ.get("INPUT_LAYER", "None"))
OUTPUT_LAYER = eval(os.environ.get("OUTPUT_LAYER", "None"))
N = Network("N")[INPUT_LAYER:OUTPUT_LAYER]
input_details = N.input_details[0]
input_lb = float(os.getenv("INPUT_LB", "-1"))
input_ub = float(os.getenv("INPUT_UB", "1"))
output_lb = float(os.getenv("OUTPUT_LB", "0.0"))

Forall(x, Implies(input_lb <= x <= input_ub, N(x) > output_lb))
