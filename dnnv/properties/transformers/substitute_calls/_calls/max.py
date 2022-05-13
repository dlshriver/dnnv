from __future__ import annotations

import numpy as np
from numpy._globals import _NoValue

from ....expressions import And, CallableExpression, Constant, Expression, IfThenElse
from .base import FunctionSubstitutor
from .utils import FunctionSubstitutionError, get_parameters


class BuiltinMax(FunctionSubstitutor):
    __matches__ = {max}

    def __call__(
        self, f: CallableExpression, *args: Expression, **kwargs: Expression
    ) -> Expression:
        if "key" in kwargs:
            raise FunctionSubstitutionError(
                "parameter 'key' of function 'max' is not currently supported"
            )
        if len(args) == 1:
            x = args[0]
            if x.is_concrete:
                if "default" in kwargs:
                    return Constant(max(x.value, default=kwargs["default"].value))
                return Constant(max(x.value))
            x_shape = x.ctx.shapes.get(x)
            if x_shape is None:
                return NotImplemented
            if x_shape == (0,):
                return kwargs["default"]
            x_indices = list(np.ndindex(*x_shape))
            new_expr: Expression = x[x_indices[-1]]
            for i in range(2, len(x_indices) + 1):
                idx = x_indices[-i]
                condition = And(
                    *(x[idx] >= x[idx2] for idx2 in x_indices[-i + 1 :]),
                )
                new_expr = IfThenElse(
                    condition,
                    x[idx],
                    new_expr,
                )
            return new_expr
        if all(arg.is_concrete for arg in args):
            return Constant(max(*(arg.value for arg in args)))
        new_expr = args[-1]
        for i in range(2, len(args) + 1):
            arg = args[-i]
            condition = And(
                *(arg >= args[-j] for j in range(1, i)),
            )
            new_expr = IfThenElse(
                condition,
                arg,
                new_expr,
            )
        return new_expr


class NumpyMax(FunctionSubstitutor):
    __matches__ = {np.amax, np.max}

    def __call__(
        self, f: CallableExpression, *args: Expression, **kwargs: Expression
    ) -> Expression:
        parameters = get_parameters(
            np.max, args, kwargs, supported_parameters={"a", "initial"}
        )
        x = parameters["a"]
        initial = parameters["initial"]
        if initial is _NoValue:
            initial = None

        assert isinstance(x, Expression)
        if x.is_concrete:
            if initial is None:
                return Constant(np.max(x.value))
            return Constant(np.max(x.value, initial=initial.value))
        x_shape = x.ctx.shapes.get(x)
        if x_shape is None:
            return NotImplemented
        if x_shape == (0,):
            return initial
        x_indices = list(np.ndindex(*x_shape))
        if initial is None:
            new_expr: Expression = x[x_indices[-1]]
        else:
            new_expr = IfThenElse(
                And(
                    x[x_indices[-1]] >= initial,
                ),
                x[x_indices[-1]],
                initial,
            )
        for i in range(2, len(x_indices) + 1):
            idx = x_indices[-i]
            if initial is None:
                condition = And(
                    *(x[idx] >= x[idx2] for idx2 in x_indices[-i + 1 :]),
                )
            else:
                condition = And(
                    x[idx] >= initial,
                    *(x[idx] >= x[idx2] for idx2 in x_indices[-i + 1 :]),
                )
            new_expr = IfThenElse(
                condition,
                x[idx],
                new_expr,
            )
        return new_expr


__all__ = ["BuiltinMax", "NumpyMax"]
