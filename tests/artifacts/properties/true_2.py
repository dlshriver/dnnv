from dnnv.properties import *
import numpy as np

N = Network()
lb = np.zeros((1, 2))
ub = np.ones((1, 2))

Forall(x, Implies(lb <= x <= ub, N(x) >= N(x)))
