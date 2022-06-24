from __future__ import annotations

from typing import Union

from ..expressions import And, IfThenElse, Implies, Not, Or
from .base import GenericExpressionTransformer


class RemoveIfThenElse(GenericExpressionTransformer):
    def __init__(self, form="dnf"):
        super().__init__()
        # `form` provides a hint on how to efficiently format the IfThenElse replacement expression
        self.form = form

    def visit_IfThenElse(self, expression: IfThenElse) -> Union[And, Or]:
        condition = self.visit(expression.condition)
        t_expr = self.visit(expression.t_expr)
        f_expr = self.visit(expression.f_expr)
        if self.form == "dnf":
            return Or(And(condition, t_expr), And(Not(condition), f_expr))
        return And(Implies(condition, t_expr), Implies(Not(condition), f_expr))

    def visit_Not(self, expression):
        form = self.form
        self.form = "cnf" if form == "dnf" else "dnf"
        result = super().generic_visit(expression)
        self.form = form
        return result


__all__ = ["RemoveIfThenElse"]
