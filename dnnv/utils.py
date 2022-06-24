from __future__ import annotations

import random
from typing import Optional, Set, Type, TypeVar

import numpy as np

T = TypeVar("T")


def get_subclasses(cls: Type[T]) -> Set[Type[T]]:
    c = list(cls.__subclasses__())
    for sub in c:
        c.extend(get_subclasses(sub))
    return set(c)


def set_random_seed(seed: Optional[int]) -> None:
    random.seed(seed)
    np.random.seed(seed)


__all__ = ["get_subclasses", "set_random_seed"]
