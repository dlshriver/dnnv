from __future__ import annotations

from typing import Callable, Union

import numpy as np

from ....expressions import (
    And,
    Call,
    Constant,
    Expression,
    IfThenElse,
    Implies,
    LogicalExpression,
    Network,
    Or,
)
from .base import FunctionSubstitutor


def argcmp_helper(cmp_fn, A, Ai, i=0) -> Union[Constant, IfThenElse]:
    if i == len(Ai) - 1:
        return Constant(i)
    cond = And(*[cmp_fn(A[Ai[i]], A[Ai[j]]) for j in range(i + 1, len(Ai))])
    return IfThenElse(cond, Constant(i), argcmp_helper(cmp_fn, A, Ai, i + 1))


def argcmp(
    cmp_fn: Callable[[Expression, Expression], Expression], A: Expression
) -> Union[Constant, IfThenElse]:
    if (
        isinstance(A, Call)
        and isinstance(A.function, Network)
        and A.function.is_concrete
    ):
        output_shape = A.function.value.output_shape[0]
        Ai = list(np.ndindex(output_shape))
        argcmp = argcmp_helper(cmp_fn, A, Ai)
        return argcmp
    return NotImplemented


def argcmp_eq(
    cmp_fn: Callable[[Expression, Expression], Expression],
    F: Call,
    E: Expression,
) -> Union[And, Or, Constant]:
    if len(F.args) > 1 or len(F.kwargs) != 0:
        raise RuntimeError("Too many arguments for argcmp")
    A = F.args[0]
    if A.is_concrete and E.is_concrete:
        A_value: np.ndarray = A.value
        assert isinstance(A_value, np.ndarray)
        j = E.value
        for i in np.ndindex(*A_value.shape):
            if not cmp_fn(A_value[j], A_value[i]):
                return Constant(False)
        return Constant(True)
    if (
        isinstance(A, Call)
        and isinstance(A.function, Network)
        and A.function.is_concrete
    ):
        output_shape = A.function.value.output_shape[0]
        Ai = list(np.ndindex(output_shape))
        if E.is_concrete:
            c = E.value
            if c >= len(Ai):
                return Constant(False)
            return And(*[cmp_fn(A[Ai[c]], A[Ai[i]]) for i in range(len(Ai)) if i != c])
        return And(*[Implies(F == c, E == c) for c in range(len(Ai))])
    return NotImplemented


class Argmax(FunctionSubstitutor):
    __matches__ = {np.argmax}

    def __call__(self, f, *args, **kwargs) -> Union[Constant, IfThenElse]:
        (A,) = args
        if A.is_concrete:
            return Constant(np.argmax(A.value))
        return argcmp(lambda a, b: a >= b, A)

    @staticmethod
    def substitute_Equal(a: Expression, b: Expression, form=None) -> LogicalExpression:
        if (
            isinstance(a, Call)
            and a.function.is_concrete
            and a.function.value == np.argmax
        ):
            return argcmp_eq(lambda x, y: x >= y, a, b)
        assert isinstance(b, Call)
        return argcmp_eq(lambda x, y: x >= y, b, a)

    @staticmethod
    def substitute_NotEqual(
        a: Expression, b: Expression, form=None
    ) -> LogicalExpression:
        result = Argmax.substitute_Equal(a, b)
        if result is NotImplemented:
            return NotImplemented
        return ~result


class Argmin(FunctionSubstitutor):
    __matches__ = {np.argmin}

    def __call__(self, f, *args, **kwargs) -> Union[Constant, IfThenElse]:
        (A,) = args
        if A.is_concrete:
            return Constant(np.argmin(A.value))
        return argcmp(lambda a, b: a <= b, *args, **kwargs)

    @staticmethod
    def substitute_Equal(a: Expression, b: Expression, form=None) -> LogicalExpression:
        if (
            isinstance(a, Call)
            and a.function.is_concrete
            and a.function.value == np.argmin
        ):
            return argcmp_eq(lambda x, y: x <= y, a, b)
        assert isinstance(b, Call)
        return argcmp_eq(lambda x, y: x <= y, b, a)

    @staticmethod
    def substitute_NotEqual(
        a: Expression, b: Expression, form=None
    ) -> LogicalExpression:
        result = Argmin.substitute_Equal(a, b)
        if result is NotImplemented:
            return NotImplemented
        return ~result


__all__ = ["Argmax", "Argmin"]
