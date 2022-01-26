from __future__ import annotations

import numpy as np

from typing import Union

from ....expressions import (
    And,
    ArithmeticExpression,
    Call,
    Constant,
    Expression,
    IfThenElse,
    Not,
    Or,
)
from .base import FunctionSubstitutor


def get_arg(expr: Call):
    assert expr.function.is_concrete
    assert expr.function.value in Abs.__matches__
    assert len(expr.args) == 1
    assert len(expr.kwargs) == 0
    return expr.args[0]


class Abs(FunctionSubstitutor):
    __matches__ = {abs, np.abs}

    def __call__(self, f, *args, **kwargs) -> Union[Constant, IfThenElse]:
        (x,) = args
        assert isinstance(x, ArithmeticExpression)
        if x.is_concrete:
            return Constant(abs(x.value))
        return IfThenElse(x >= 0, x, -x)

    @staticmethod
    def substitute_Equal(a: Expression, b: Expression) -> Expression:
        if (
            isinstance(a, Call)
            and a.function.is_concrete
            and a.function.value in Abs.__matches__
        ):
            a = get_arg(a)
            return And(b >= 0, Or(And(a >= 0, a == b), And(a < 0, -a == b)))
        b = get_arg(b)
        return And(a >= 0, Or(And(b >= 0, a == b), And(b < 0, a == -b)))

    @staticmethod
    def substitute_GreaterThan(a: Expression, b: Expression) -> Expression:
        if (
            isinstance(a, Call)
            and a.function.is_concrete
            and a.function.value in Abs.__matches__
        ):
            a = get_arg(a)
            return Or(Not(b >= 0), And(a >= 0, a > b), And(a < 0, -a > b))
        b = get_arg(b)
        return Or(Not(a >= 0), And(b >= 0, a > b), And(b < 0, a > -b))

    @staticmethod
    def substitute_GreaterThanOrEqual(a: Expression, b: Expression) -> Expression:
        if (
            isinstance(a, Call)
            and a.function.is_concrete
            and a.function.value in Abs.__matches__
        ):
            a = get_arg(a)
            return Or(Not(b >= 0), And(a >= 0, a >= b), And(a < 0, -a >= b))
        b = get_arg(b)
        return Or(Not(a >= 0), And(b >= 0, a >= b), And(b < 0, a >= -b))

    @staticmethod
    def substitute_LessThan(a: Expression, b: Expression) -> Expression:
        if (
            isinstance(a, Call)
            and a.function.is_concrete
            and a.function.value in Abs.__matches__
        ):
            a = get_arg(a)
            return And(b >= 0, Or(And(a >= 0, a < b), And(a < 0, -a < b)))
        b = get_arg(b)
        return And(a >= 0, Or(And(b >= 0, a < b), And(b < 0, a < -b)))

    @staticmethod
    def substitute_LessThanOrEqual(a: Expression, b: Expression) -> Expression:
        if (
            isinstance(a, Call)
            and a.function.is_concrete
            and a.function.value in Abs.__matches__
        ):
            a = get_arg(a)
            return And(b >= 0, Or(And(a >= 0, a <= b), And(a < 0, -a <= b)))
        b = get_arg(b)
        return And(a >= 0, Or(And(b >= 0, a <= b), And(b < 0, a <= -b)))

    @staticmethod
    def substitute_NotEqual(a: Expression, b: Expression) -> Expression:
        if (
            isinstance(a, Call)
            and a.function.is_concrete
            and a.function.value in Abs.__matches__
        ):
            a = get_arg(a)
            return Or(b < 0, And(a >= 0, a != b), And(a < 0, -a != b))
        b = get_arg(b)
        return Or(a < 0, And(b >= 0, a != b), And(b < 0, a != -b))


__all__ = ["Abs"]
