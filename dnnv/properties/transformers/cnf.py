from __future__ import annotations

from typing import Optional, Set, Tuple, Union

from ..expressions import *
from ..visitors.inference import DetailsInference
from .base import GenericExpressionTransformer
from .lift_ifthenelse import LiftIfThenElse
from .remove_ifthenelse import RemoveIfThenElse
from .substitute_calls import SubstituteCalls


class CnfTransformer(GenericExpressionTransformer):
    def visit(self, expression: Expression) -> Union[And, Constant]:
        if self._top_level:
            expression = expression.propagate_constants()
            DetailsInference().visit(expression)
            expression = SubstituteCalls(form="cnf").visit(expression)
            expression = expression.propagate_constants()
            expression = LiftIfThenElse().visit(expression)
            expression = expression.propagate_constants()
            expression = RemoveIfThenElse(form="cnf").visit(expression)
            expression = expression.propagate_constants()
            if not isinstance(expression, ArithmeticExpression) or isinstance(
                expression, Symbol
            ):
                expression = And(Or(expression))
        expr = super().visit(expression)
        return expr

    def visit_And(self, expression: And) -> Union[And, Constant]:
        expressions: Set[Expression] = set()
        for expr in expression.expressions:
            expr = self.visit(expr)
            if isinstance(expr, And):
                expressions = expressions.union(expr.expressions)
            else:
                expressions.add(Or(expr))
        return And(*expressions)

    def visit_Exists(self, expression: Exists) -> Union[And, Constant]:
        raise NotImplementedError("Skolemization is not yet implemented.")

    def visit_Forall(self, expression: Forall) -> Union[And, Constant]:
        expr = self.visit(expression.expression)
        return expr

    def visit_Implies(self, expression: Implies) -> Union[And, Constant]:
        assert isinstance(expression.expr1, LogicalExpression)
        return self.visit(Or(~expression.expr1, expression.expr2))

    def visit_Or(self, expression: Or) -> Union[And, Constant]:
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
