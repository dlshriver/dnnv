from __future__ import annotations

import inspect

import numpy as np
from numpy._globals import _NoValue

from ....expressions import Add, CallableExpression, Constant, Expression
from .base import FunctionSubstitutor
from .utils import get_parameters


class BuiltinSum(FunctionSubstitutor):
    __matches__ = {sum}

    def __call__(
        self, f: CallableExpression, *args: Expression, **kwargs: Expression
    ) -> Expression:
        x = args[0]
        if len(args) > 1:
            start = args[1]
        else:
            start = kwargs.get("start", Constant(0))

        assert isinstance(x, Expression)
        if x.is_concrete:
            return Constant(sum(x.value, start.value))
        x_shape = x.ctx.shapes.get(x)
        if x_shape is None:
            return NotImplemented
        return Add(start, *(x[idx] for idx in np.ndindex(*x_shape)))


class NumpySum(FunctionSubstitutor):
    __matches__ = {np.sum}
    sig = inspect.signature(np.sum)

    def __call__(
        self, f: CallableExpression, *args: Expression, **kwargs: Expression
    ) -> Expression:
        parameters = get_parameters(
            np.sum, args, kwargs, supported_parameters={"a", "initial"}
        )
        x = parameters["a"]
        initial = parameters["initial"]
        if initial is _NoValue:
            initial = None

        assert isinstance(x, Expression)
        if x.is_concrete:
            if initial is None:
                return Constant(np.sum(x.value))
            return Constant(np.sum(x.value, initial=initial.value))
        x_shape = x.ctx.shapes.get(x)
        if x_shape is None:
            return NotImplemented
        if x_shape == ():
            if initial is None:
                return x
            return Add(x, initial)
        if initial is None:
            return Add(*(x[idx] for idx in np.ndindex(*x_shape)))
        return Add(initial, *(x[idx] for idx in np.ndindex(*x_shape)))


__all__ = ["BuiltinSum", "NumpySum"]
