from __future__ import annotations

import numpy as np

from typing import Union

from ....expressions import ArithmeticExpression, Constant, Expression, IfThenElse
from .base import FunctionSubstitutor


class Abs(FunctionSubstitutor):
    __matches__ = {abs, np.abs}

    def __call__(self, f, *args, **kwargs) -> Union[Constant, IfThenElse]:
        (x,) = args
        assert isinstance(x, ArithmeticExpression)
        if x.is_concrete:
            return Constant(abs(x.value))
        return IfThenElse(x >= 0, x, -x)


__all__ = ["Abs"]
