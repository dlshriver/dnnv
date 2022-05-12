import numpy as np

from dnnv.properties import *

N = Network("N")
ub = np.ones((1, 2))

Forall(x, Implies(abs(x) <= ub, N(x) >= 1))
