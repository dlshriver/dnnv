from __future__ import annotations

import numpy as np

from .... import Expression, expressions
from .base import FunctionSubstitutor


class Shape(FunctionSubstitutor):
    __matches__ = {len, np.shape}

    def __call__(
        self, f: expressions.CallableExpression, *args: Expression, **kwargs: Expression
    ) -> Expression:
        assert len(args) == 1
        assert len(kwargs) == 0
        (x,) = args
        if x not in x.ctx.shapes:
            return NotImplemented
        shape = x.ctx.shapes[x]
        if f.value == len:
            return expressions.Constant(shape[0])
        return expressions.Constant(shape)


__all__ = ["Shape"]
