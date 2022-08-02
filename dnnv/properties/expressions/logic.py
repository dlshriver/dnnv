from __future__ import annotations

import operator
import typing
from typing import Optional, Type, TypeVar

import numpy as np

from ..errors import DNNVExpressionError, NonConcreteExpressionError
from .base import AssociativeExpression, BinaryExpression, Expression, UnaryExpression
from .context import Context

if typing.TYPE_CHECKING:  # pragma: no cover
    from .terms import Symbol

NonUnaryLogicalOperatorType = TypeVar("NonUnaryLogicalOperatorType", "And", "Or")


class LogicalExpression(Expression):
    def __operator__(
        self, other, operator_type: Type[NonUnaryLogicalOperatorType]
    ) -> NonUnaryLogicalOperatorType:
        if isinstance(other, LogicalExpression):
            return operator_type(self, other)
        elif not isinstance(other, Expression):
            from .terms import Constant

            return operator_type(self, Constant(other))
        return NotImplemented

    def __roperator__(
        self, other, operator_type: Type[NonUnaryLogicalOperatorType]
    ) -> NonUnaryLogicalOperatorType:
        if not isinstance(other, Expression):
            from .terms import Constant

            return operator_type(Constant(other), self)
        return NotImplemented

    def __and__(self, other) -> And:
        return self.__operator__(other, And)

    def __invert__(self):
        return Not(self)

    def __or__(self, other) -> Or:
        return self.__operator__(other, Or)

    def __rand__(self, other) -> And:
        return self.__roperator__(other, And)

    def __ror__(self, other) -> Or:
        return self.__roperator__(other, Or)


class Not(LogicalExpression, UnaryExpression):
    OPERATOR_SYMBOL = "~"

    def OPERATOR(self, expr):
        if isinstance(expr, np.ndarray):
            return ~expr
        return not expr

    def __invert__(self):
        return self.expr


class Or(LogicalExpression, AssociativeExpression):
    BASE_VALUE = False
    DOMINANT_VALUES = [True]
    OPERATOR = operator.or_
    OPERATOR_SYMBOL = "|"

    def __init__(self, *expr: Expression, ctx: Optional[Context] = None):
        expr_set = set()
        expressions = []
        for e in expr:
            if e in expr_set:
                continue
            expr_set.add(e)
            expressions.append(e)
        super().__init__(*expressions, ctx=ctx)

    def __invert__(self):
        return And(*[~expr for expr in self.expressions], ctx=self.ctx)


class And(LogicalExpression, AssociativeExpression):
    BASE_VALUE = True
    DOMINANT_VALUES = [False]
    OPERATOR = operator.and_
    OPERATOR_SYMBOL = "&"

    def __init__(self, *expr: Expression, ctx: Optional[Context] = None):
        expr_set = set()
        expressions = []
        for e in expr:
            if e in expr_set:
                continue
            expr_set.add(e)
            expressions.append(e)
        super().__init__(*expressions, ctx=ctx)

    def __invert__(self):
        return Or(*[~expr for expr in self.expressions], ctx=self.ctx)


class Implies(LogicalExpression, BinaryExpression):
    OPERATOR_SYMBOL = "==>"

    def OPERATOR(self, a, b):
        return operator.or_(operator.xor(True, a), b)

    def __invert__(self):
        return And(self.expr1, ~self.expr2, ctx=self.ctx)


class Quantifier(LogicalExpression):
    def __init__(
        self,
        variable: Symbol,
        formula: Expression,
        *,
        ctx: Optional[Context] = None,
    ):
        super().__init__(ctx=ctx)
        if variable.is_concrete:
            raise DNNVExpressionError(
                "Quantifier variable should be symbolic, not concrete."
                f" Got '{variable}={variable.value}'."
            )
        self.variable = variable
        self.expression = formula

    def __repr__(self):
        return f"{type(self).__name__}({self.variable!r}, {self.expression!r})"

    def __str__(self):
        return f"{type(self).__name__}({self.variable}, {self.expression})"

    def is_equivalent(self, other) -> bool:
        if super().is_equivalent(other):
            return True
        if (
            type(self) == type(other)
            and self.variable.is_equivalent(other.variable)
            and self.expression.is_equivalent(other.expression)
        ):
            return True
        return False


class Forall(Quantifier):
    def __invert__(self):
        return Exists(self.variable, ~self.expression, ctx=self.ctx)


class Exists(Quantifier):
    def __invert__(self):
        return Forall(self.variable, ~self.expression, ctx=self.ctx)


## Comparisons


class Equal(LogicalExpression, BinaryExpression):
    OPERATOR = operator.eq
    OPERATOR_SYMBOL = "=="

    def __invert__(self) -> Not:
        return Not(self, ctx=self.ctx)

    def __bool__(self):
        if self.expr1 is self.expr2:
            return True
        try:
            return bool(np.all(self.expr1.value == self.expr2.value))
        except NonConcreteExpressionError:
            return self.expr1.is_equivalent(self.expr2)


class GreaterThan(LogicalExpression, BinaryExpression):
    OPERATOR = operator.gt
    OPERATOR_SYMBOL = ">"

    def __invert__(self) -> Not:
        return Not(self, ctx=self.ctx)


class GreaterThanOrEqual(LogicalExpression, BinaryExpression):
    OPERATOR = operator.ge
    OPERATOR_SYMBOL = ">="

    def __invert__(self) -> Not:
        return Not(self, ctx=self.ctx)


class LessThan(LogicalExpression, BinaryExpression):
    OPERATOR = operator.lt
    OPERATOR_SYMBOL = "<"

    def __invert__(self) -> Not:
        return Not(self, ctx=self.ctx)


class LessThanOrEqual(LogicalExpression, BinaryExpression):
    OPERATOR = operator.le
    OPERATOR_SYMBOL = "<="

    def __invert__(self) -> Not:
        return Not(self, ctx=self.ctx)


class NotEqual(LogicalExpression, BinaryExpression):
    OPERATOR = operator.ne
    OPERATOR_SYMBOL = "!="

    def __invert__(self) -> Not:
        return Not(self, ctx=self.ctx)

    def __bool__(self):
        try:
            return bool(np.any(self.expr1.value != self.expr2.value))
        except NonConcreteExpressionError:
            return not self.expr1.is_equivalent(self.expr2)


__all__ = [
    # Base
    "And",
    "Exists",
    "Forall",
    "Implies",
    "LogicalExpression",
    "Not",
    "Or",
    "Quantifier",
    # Comparisons
    "Equal",
    "GreaterThan",
    "GreaterThanOrEqual",
    "LessThan",
    "LessThanOrEqual",
    "NotEqual",
]
