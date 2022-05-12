import numpy as np

from dnnv.properties import *

N = Network("N")
lb = np.zeros((1, 2))
ub = np.ones((1, 2))

Forall(x, Implies(lb <= x <= ub, N(x) >= N(x)))
