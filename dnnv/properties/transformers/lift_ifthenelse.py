from __future__ import annotations

from typing import List, Union

from ..expressions import (
    AssociativeExpression,
    BinaryExpression,
    Expression,
    IfThenElse,
    UnaryExpression,
)
from .base import GenericExpressionTransformer


class LiftIfThenElse(GenericExpressionTransformer):
    def generic_visit(self, expression: Expression) -> Expression:
        if isinstance(expression, AssociativeExpression):
            return self.visit_AssociativeExpression(expression)
        elif isinstance(expression, BinaryExpression):
            return self.visit_BinaryExpression(expression)
        elif isinstance(expression, UnaryExpression):
            return self.visit_UnaryExpression(expression)
        return super().generic_visit(expression)

    def visit_AssociativeExpression(
        self,
        expression: AssociativeExpression,
    ) -> Union[AssociativeExpression, IfThenElse]:
        expr_t = type(expression)
        exprs = [self.visit(expr) for expr in expression.expressions]
        if all(not isinstance(expr, IfThenElse) for expr in exprs):
            return expr_t(*exprs)
        expressions: List[Expression] = []
        ite_exprs: List[IfThenElse] = []
        for expr in exprs:
            if isinstance(expr, IfThenElse):
                ite_exprs.append(expr)
            else:
                expressions.append(expr)
        if len(ite_exprs) > 1:
            t_expr = self.visit(
                expr_t(ite_exprs[0].t_expr, *ite_exprs[1:], *expressions)
            )
            f_expr = self.visit(
                expr_t(ite_exprs[0].f_expr, *ite_exprs[1:], *expressions)
            )
        else:
            t_expr = self.visit(expr_t(ite_exprs[0].t_expr, *expressions))
            f_expr = self.visit(expr_t(ite_exprs[0].f_expr, *expressions))
        return IfThenElse(ite_exprs[0].condition, t_expr, f_expr)

    def visit_BinaryExpression(
        self,
        expression: BinaryExpression,
    ) -> Union[BinaryExpression, IfThenElse]:
        expr_t = type(expression)
        expr1 = self.visit(expression.expr1)
        expr2 = self.visit(expression.expr2)
        if not isinstance(expr1, IfThenElse) and not isinstance(expr2, IfThenElse):
            return expr_t(expr1, expr2)
        if isinstance(expr1, IfThenElse):
            if isinstance(expr1.t_expr, IfThenElse) or isinstance(expr2, IfThenElse):
                t_expr = self.visit(expr_t(expr1.t_expr, expr2))
            else:
                t_expr = expr_t(expr1.t_expr, expr2)
            if isinstance(expr1.f_expr, IfThenElse) or isinstance(expr2, IfThenElse):
                f_expr = self.visit(expr_t(expr1.f_expr, expr2))
            else:
                f_expr = expr_t(expr1.f_expr, expr2)
            return IfThenElse(expr1.condition, t_expr, f_expr)
        assert isinstance(expr2, IfThenElse)
        if isinstance(expr1, IfThenElse) or isinstance(expr2.t_expr, IfThenElse):
            t_expr = self.visit(expr_t(expr1, expr2.t_expr))
        else:
            t_expr = expr_t(expr1, expr2.t_expr)
        if isinstance(expr1, IfThenElse) or isinstance(expr2.f_expr, IfThenElse):
            f_expr = self.visit(expr_t(expr1, expr2.f_expr))
        else:
            f_expr = expr_t(expr1, expr2.f_expr)
        return IfThenElse(expr2.condition, t_expr, f_expr)

    def visit_UnaryExpression(
        self,
        expression: UnaryExpression,
    ) -> Union[IfThenElse, UnaryExpression]:
        expr_type = type(expression)
        expr = self.visit(expression.expr)
        if not isinstance(expr, IfThenElse):
            return expr_type(expr)
        if isinstance(expr.t_expr, IfThenElse):
            t_expr = self.visit_UnaryExpression(expr_type(expr.t_expr))
        else:
            t_expr = expr_type(expr.t_expr)
        if isinstance(expr.f_expr, IfThenElse):
            f_expr = self.visit_UnaryExpression(expr_type(expr.f_expr))
        else:
            f_expr = expr_type(expr.f_expr)
        return IfThenElse(expr.condition, t_expr, f_expr)


__all__ = ["LiftIfThenElse"]
