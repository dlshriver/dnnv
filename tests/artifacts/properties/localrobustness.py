from dnnv.properties import *
import numpy as np

N = Network()
x = np.random.randn(1, 2) + 5

Forall(x_, Implies(x - 0.1 <= x_ <= x + 0.1, np.argmax(N(x_)) == np.argmax(N(x))))