import numpy as np
import random
import sys


def get_subclasses(cls):
    c = list(cls.__subclasses__())
    for sub in c:
        c.extend(get_subclasses(sub))
    return set(c)


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
