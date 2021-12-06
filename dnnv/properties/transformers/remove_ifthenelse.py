from __future__ import annotations

from typing import List, Union

from ..expressions import And, IfThenElse, Implies
from .base import GenericExpressionTransformer


class RemoveIfThenElse(GenericExpressionTransformer):
    def visit_IfThenElse(self, expression: IfThenElse) -> And:
        condition = self.visit(expression.condition)
        t_expr = self.visit(expression.t_expr)
        f_expr = self.visit(expression.f_expr)
        return And(Implies(condition, t_expr), Implies(~condition, f_expr))


__all__ = ["RemoveIfThenElse"]
