from __future__ import annotations

from ..expressions import *
from .base import GenericExpressionTransformer
from .substitute_expression import SubstituteExpression


class FirstOrderLogicTransformer(GenericExpressionTransformer):
    def __init__(self, form="dnf"):
        super().__init__(recursive=True)
        self.form = form

    def visit(self, expression: Expression) -> Expression:
        expr = super().visit(expression)
        return expr

    def visit_Finally(self, expression: ltl.Finally) -> Expression:
        result = self.visit(ltl.Until(Constant(True), expression.expr))
        return result

    def visit_Globally(self, expression: ltl.Globally) -> Expression:
        return Not(self.visit(ltl.Until(Constant(True), Not(expression.expr))))

    def visit_Next(self, expression: ltl.Next) -> Expression:
        variables = set(
            e for e in expression.iter() if isinstance(e, ltl.TemporalSymbol)
        )
        expr = expression.expr
        if len(variables) == 0:
            return expr
        try:
            next_variables = [v.next() for v in variables]
        except ltl.LTLMaxStepReached:
            return Constant(False)
        for v, next_v in zip(variables, next_variables):
            expr = SubstituteExpression(v, next_v).visit(expr)
        return self.visit(expr)

    def visit_Release(self, expression: ltl.Release) -> Expression:
        return Not(self.visit(ltl.Until(Not(expression.expr1), Not(expression.expr2))))

    def visit_StrongRelease(self, expression: ltl.StrongRelease) -> Expression:
        return self.visit(
            ltl.Until(expression.expr2, And(expression.expr1, expression.expr2))
        )

    def visit_TemporalSymbol(self, expression: ltl.TemporalSymbol) -> Expression:
        return expression.freeze()

    def visit_Until(self, expression: ltl.Until) -> Expression:
        if expression.expr1 is Constant(True):
            result = Or(
                self.visit(expression.expr2),
                self.visit(ltl.Next(ltl.Until(expression.expr1, expression.expr2))),
            )
        else:
            result = Or(
                self.visit(expression.expr2),
                And(
                    self.visit(expression.expr1),
                    self.visit(ltl.Next(ltl.Until(expression.expr1, expression.expr2))),
                ),
            )
        return result

    def visit_WeakUntil(self, expression: ltl.WeakUntil) -> Expression:
        return Or(
            self.visit(ltl.Until(expression.expr1, expression.expr2)),
            Not(self.visit(ltl.Until(Constant(True), Not(expression.expr1)))),
        )


__all__ = ["FirstOrderLogicTransformer"]
