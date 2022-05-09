from __future__ import annotations

from .arithmetic import ArithmeticExpression
from .base import BinaryExpression
from .call import CallableExpression
from .logic import LogicalExpression


class Attribute(
    ArithmeticExpression,
    CallableExpression,
    LogicalExpression,
    BinaryExpression,
):
    def OPERATOR(self, expr, name):
        try:
            return getattr(expr, name)
        except AttributeError as e:
            raise ValueError(e)

    @property
    def expr(self):
        return self.expr1

    @property
    def name(self):
        return self.expr2

    def __repr__(self):
        return f"{self.expr!r}.{self.name!r}"

    def __str__(self):
        return f"{self.expr}.{self.name}"


__all__ = ["Attribute"]
