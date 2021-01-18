import numpy as np
import random
import sys

from typing import Optional, Set, Type, TypeVar

T = TypeVar("T")


def get_subclasses(cls: Type[T]) -> Set[Type[T]]:
    c = list(cls.__subclasses__())
    for sub in c:
        c.extend(get_subclasses(sub))
    return set(c)


def set_random_seed(seed: Optional[int]) -> None:
    random.seed(seed)
    np.random.seed(seed)
