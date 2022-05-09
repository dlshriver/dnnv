from __future__ import annotations

from .arithmetic import ArithmeticExpression
from .base import TernaryExpression
from .call import CallableExpression
from .logic import LogicalExpression


class IfThenElse(
    ArithmeticExpression,
    CallableExpression,
    LogicalExpression,
    TernaryExpression,
):
    def OPERATOR(self, cond, t_expr, f_expr):
        if cond:
            return t_expr
        return f_expr

    @property
    def condition(self):
        return self.expr1

    @property
    def t_expr(self):
        return self.expr2

    @property
    def f_expr(self):
        return self.expr3


__all__ = ["IfThenElse"]
