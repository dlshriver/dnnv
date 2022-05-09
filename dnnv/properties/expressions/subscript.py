from __future__ import annotations

from .arithmetic import ArithmeticExpression
from .base import BinaryExpression
from .call import CallableExpression
from .logic import LogicalExpression


class Subscript(
    ArithmeticExpression,
    CallableExpression,
    LogicalExpression,
    BinaryExpression,
):
    def OPERATOR(self, expr, index):
        return expr[index]

    @property
    def expr(self):
        return self.expr1

    @property
    def index(self):
        return self.expr2

    def __repr__(self):
        return f"{self.expr!r}[{self.index!r}]"

    def __str__(self):
        return f"{self.expr}[{self.index}]"


__all__ = ["Subscript"]
