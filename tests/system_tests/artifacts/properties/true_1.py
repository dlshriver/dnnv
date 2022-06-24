import numpy as np

from dnnv.properties import *

N = Network("N")

Forall(x, Implies(False, N(x) > 0))
