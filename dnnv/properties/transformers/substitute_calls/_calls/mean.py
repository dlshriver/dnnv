from __future__ import annotations

import numpy as np

from ....expressions import Add, CallableExpression, Constant, Expression
from .base import FunctionSubstitutor
from .utils import get_parameters


class NumpyMean(FunctionSubstitutor):
    __matches__ = {np.mean}

    def __call__(
        self, f: CallableExpression, *args: Expression, **kwargs: Expression
    ) -> Expression:
        parameters = get_parameters(
            np.mean, args, kwargs, supported_parameters={"a", "initial"}
        )
        x = parameters["a"]

        assert isinstance(x, Expression)
        if x.is_concrete:
            return Constant(np.mean(x.value))
        x_shape = x.ctx.shapes.get(x)
        if x_shape is None:
            return NotImplemented
        if x_shape == ():
            return x
        return Add(*(x[idx] for idx in np.ndindex(*x_shape))) / Constant(
            np.product(x_shape)
        )


__all__ = ["NumpyMean"]
