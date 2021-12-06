from __future__ import annotations

from typing import Optional, Set, Tuple

from .base import GenericExpressionTransformer
from .lift_ifthenelse import LiftIfThenElse
from .remove_ifthenelse import RemoveIfThenElse
from ..expressions import *


class CnfTransformer(GenericExpressionTransformer):
    def visit(self, expression: Expression) -> Expression:
        if self._top_level:
            expression = expression.propagate_constants()
            expression = LiftIfThenElse().visit(expression)
            expression = expression.propagate_constants()
            expression = RemoveIfThenElse().visit(expression)
            expression = expression.propagate_constants()
            expression = And(Or(expression))
        expr = super().visit(expression)
        return expr

    def visit_And(self, expression: And) -> And:
        expressions: Set[Expression] = set()
        for expr in expression.expressions:
            expr = self.visit(expr)
            if isinstance(expr, And):
                expressions = expressions.union(expr.expressions)
            else:
                if not isinstance(expr, Or):
                    expr = Or(expr)
                expressions.add(expr)
        return And(*expressions)

    def visit_Exists(self, expression: Exists):
        raise NotImplementedError("Skolemization is not yet implemented.")

    def visit_Forall(self, expression: Forall):
        expr = self.visit(expression.expression)
        return expr

    def visit_Implies(self, expression: Implies) -> And:
        return self.visit(Or(~expression.expr1, expression.expr2))

    def visit_Or(self, expression: Or) -> And:
        conjunction: Optional[And] = None
        expressions: Set[Expression] = set()
        for expr in expression.expressions:
            expr = self.visit(expr)
            if conjunction is None and isinstance(expr, And):
                conjunction = expr
            else:
                expressions.add(expr)
        if conjunction is None:
            if len(expressions) == 0:
                return And(Or(Constant(False)))
            return And(Or(*expressions))
        elif len(expressions) == 0:
            return conjunction
        clauses: Set[Tuple[Expression, ...]] = set(
            (e,) for e in conjunction.expressions
        )
        for expr in expressions:
            if isinstance(expr, And):
                new_clauses = set()
                for clause in clauses:
                    for e in expr.expressions:
                        new_clauses.add(tuple(set(clause + (e,))))
                clauses = new_clauses
            else:
                assert not isinstance(expr, Or)
                new_clauses = set()
                for clause in clauses:
                    new_clauses.add(tuple(set(clause + (expr,))))
                clauses = new_clauses
        return And(*[Or(*clause) for clause in clauses])


__all__ = ["CnfTransformer"]
