from __future__ import annotations

from .base import GenericExpressionTransformer
from .substitute_expression import SubstituteExpression
from ..expressions import *


class FirstOrderLogicTransformer(GenericExpressionTransformer):
    def visit(self, expression: Expression) -> Expression:
        expr = super().visit(expression)
        return expr

    def visit_Finally(self, expression: ltl.Finally) -> Expression:
        return self.visit(ltl.Until(Constant(True), expression.expr))

    def visit_Globally(self, expression: ltl.Globally) -> Expression:
        return self.visit(ltl.Release(Constant(False), expression.expr))

    def visit_Next(self, expression: ltl.Next) -> Expression:
        variables = set(
            e for e in expression.iter() if isinstance(e, ltl.TemporalSymbol)
        )
        expr = expression.expr
        if len(variables) == 0:
            return expr
        for v in variables:
            try:
                next_v = v.next()
            except ltl.LTLUnwindingBoundReached:
                return Constant(False)
            expr = SubstituteExpression(v, next_v).visit(expr)
        return self.visit(expr)

    def visit_Release(self, expression: ltl.Release) -> Expression:
        return self.visit(
            ltl.WeakUntil(expression.expr2, And(expression.expr1, expression.expr2))
        )

    def visit_StrongRelease(self, expression: ltl.StrongRelease) -> Expression:
        return self.visit(
            ltl.Until(expression.expr2, And(expression.expr1, expression.expr2))
        )

    def visit_TemporalSymbol(self, expression: ltl.TemporalSymbol) -> Expression:
        return expression.freeze()

    def visit_Until(self, expression: ltl.Until) -> Expression:
        return self.visit(
            Or(
                expression.expr2,
                And(
                    expression.expr1,
                    ltl.Next(ltl.Until(expression.expr1, expression.expr2)),
                ),
            )
        )

    def visit_WeakUntil(self, expression: ltl.WeakUntil) -> Expression:
        return self.visit(
            Or(
                ltl.Until(expression.expr1, expression.expr2),
                # ltl.Globally(expression.expr1),
                Not(ltl.Until(Constant(True), Not(expression.expr1))),
            )
        )


__all__ = ["FirstOrderLogicTransformer"]
