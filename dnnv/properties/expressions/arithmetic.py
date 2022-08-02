from __future__ import annotations

import operator
from typing import Type, TypeVar

from .base import AssociativeExpression, BinaryExpression, Expression, UnaryExpression

NonUnaryArithmeticType = TypeVar(
    "NonUnaryArithmeticType", "Add", "Divide", "Multiply", "Subtract"
)


class ArithmeticExpression(Expression):
    def __arithmetic__(
        self, other, arithmetic_type: Type[NonUnaryArithmeticType]
    ) -> NonUnaryArithmeticType:
        if isinstance(other, ArithmeticExpression):
            return arithmetic_type(self, other)
        elif not isinstance(other, Expression):
            from .terms import Constant

            return arithmetic_type(self, Constant(other))
        return NotImplemented

    def __rarithmetic__(
        self, other, arithmetic_type: Type[NonUnaryArithmeticType]
    ) -> NonUnaryArithmeticType:
        if not isinstance(other, Expression):
            from .terms import Constant

            return arithmetic_type(Constant(other), self)
        return NotImplemented

    def __add__(self, other) -> Add:
        return self.__arithmetic__(other, Add)

    def __mul__(self, other) -> Multiply:
        return self.__arithmetic__(other, Multiply)

    def __neg__(self) -> Negation:
        return Negation(self)

    def __sub__(self, other) -> Subtract:
        return self.__arithmetic__(other, Subtract)

    def __radd__(self, other) -> Add:
        return self.__rarithmetic__(other, Add)

    def __rmul__(self, other) -> Multiply:
        return self.__rarithmetic__(other, Multiply)

    def __rsub__(self, other) -> Subtract:
        return self.__rarithmetic__(other, Subtract)

    def __rtruediv__(self, other) -> Divide:
        return self.__rarithmetic__(other, Divide)

    def __truediv__(self, other) -> Divide:
        return self.__arithmetic__(other, Divide)


class Negation(ArithmeticExpression, UnaryExpression):
    OPERATOR = operator.neg
    OPERATOR_SYMBOL = "-"

    def __neg__(self):
        return self.expr


class Add(ArithmeticExpression, AssociativeExpression):
    BASE_VALUE = 0
    OPERATOR = operator.add
    OPERATOR_SYMBOL = "+"


class Subtract(ArithmeticExpression, BinaryExpression):
    OPERATOR = operator.sub
    OPERATOR_SYMBOL = "-"


class Multiply(ArithmeticExpression, AssociativeExpression):
    BASE_VALUE = 1
    DOMINANT_VALUES = [0]
    OPERATOR = operator.mul
    OPERATOR_SYMBOL = "*"


class Divide(ArithmeticExpression, BinaryExpression):
    OPERATOR = operator.truediv
    OPERATOR_SYMBOL = "/"


__all__ = [
    "Add",
    "ArithmeticExpression",
    "Divide",
    "Multiply",
    "Negation",
    "Subtract",
]
