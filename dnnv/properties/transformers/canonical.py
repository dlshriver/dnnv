from __future__ import annotations

import numpy as np

from collections import defaultdict
from typing import Dict, List, Tuple

from .dnf import DnfTransformer
from .substitute_calls import SubstituteCalls
from ..expressions import *


def _extract_constants(expr: Add) -> Tuple[Add, Constant]:
    constants = []
    summands = []
    for e in expr.expressions:
        if e.is_concrete:
            constants.append(e)
        else:
            summands.append(e)
    return Add(*summands), Constant(-Add(*constants).value)


class CanonicalTransformer(DnfTransformer):
    def visit(self, expression: Expression) -> Expression:
        if self._top_level:
            expression = expression.propagate_constants()
            expression = SubstituteCalls().visit(expression)
        expr = super().visit(expression)
        return expr

    def _extract_coefficients(self, expression: Add) -> Dict[Expression, Expression]:
        coefficients = defaultdict(lambda: Constant(0))
        for expr in expression.expressions:
            expr = self.visit(expr)
            if isinstance(expr, Add):
                if len(expr.expressions) != 1:
                    for expr, coeff in self._extract_coefficients(expr).items():
                        coefficients[expr] = coefficients[expr] + coeff
                    continue
                expr = expr.expressions[0]
            value = 1
            if expr.is_concrete:
                expr, value = Constant(value), expr.value
            elif isinstance(expr, Multiply):
                constants = []
                symbols = []
                for e in expr.expressions:
                    if e.is_concrete:
                        constants.append(e)
                    else:
                        symbols.append(e)
                if len(symbols) == 1:
                    expr = symbols[0]
                else:
                    expr = Multiply(*symbols)
                value = Multiply(*constants).value
            coefficients[expr] = coefficients[expr] + Constant(value)
        return coefficients

    def visit_Add(self, expression: Add) -> Add:
        coefficients = self._extract_coefficients(expression)
        summands = []
        for expr, coefficient in coefficients.items():
            coeff = coefficient.value
            if np.all(coeff == 0):
                continue
            summands.append(Multiply(Constant(coeff), expr))
        return Add(*summands)

    def visit_Equal(self, expression: Equal) -> Or:
        expr1 = expression.expr1
        expr2 = expression.expr2
        return self.visit(Or(And(expr1 <= expr2, expr2 <= expr1)))

    def visit_GreaterThan(self, expression: GreaterThan) -> Or:
        lhs, rhs = _extract_constants(
            self.visit(Add(Multiply(Constant(-1), expression.expr1), expression.expr2))
        )
        expr = Or(And(LessThan(lhs, rhs)))
        return expr

    def visit_GreaterThanOrEqual(self, expression: GreaterThanOrEqual) -> Or:
        lhs, rhs = _extract_constants(
            self.visit(Add(Multiply(Constant(-1), expression.expr1), expression.expr2))
        )
        expr = Or(And(LessThanOrEqual(lhs, rhs)))
        return expr

    def visit_LessThan(self, expression: LessThan) -> Or:
        lhs, rhs = _extract_constants(
            self.visit(Add(expression.expr1, Multiply(Constant(-1), expression.expr2)))
        )
        expr = Or(And(LessThan(lhs, rhs)))
        return expr

    def visit_LessThanOrEqual(self, expression: LessThanOrEqual) -> Or:
        lhs, rhs = _extract_constants(
            self.visit(Add(expression.expr1, Multiply(Constant(-1), expression.expr2)))
        )
        expr = Or(And(LessThanOrEqual(lhs, rhs)))
        return expr

    def visit_Multiply(self, expression: Multiply) -> Add:
        expressions: List[List[Expression]] = [[]]
        for expr in expression.expressions:
            expr = self.visit(expr)
            if isinstance(expr, Add):
                new_expressions = []
                for e in expressions:
                    for e_ in expr.expressions:
                        new_e = [v for v in e]
                        new_e.append(e_)
                        new_expressions.append(new_e)
                expressions = new_expressions
            else:
                for e in expressions:
                    e.append(expr)
        if len(expressions) <= 1:
            consts = []
            symbols = [Constant(1)]
            for e in expressions[0]:
                if e.is_concrete:
                    consts.append(e)
                else:
                    symbols.append(e)
            const = Constant(Multiply(*consts).value)
            product = Multiply(*symbols).propagate_constants()
            return Add(Multiply(const, product))
        return self.visit(Add(*[Multiply(*e) for e in expressions]))

    def visit_Negation(self, expression: Negation) -> Add:
        return self.visit(Multiply(Constant(-1), expression.expr))

    def visit_NotEqual(self, expression: NotEqual) -> Or:
        expr1 = expression.expr1
        expr2 = expression.expr2
        return self.visit(Or(And(expr1 > expr2), And(expr2 > expr1)))

    def visit_Subtract(self, expression: Subtract) -> Add:
        return self.visit(
            Add(expression.expr1, Multiply(Constant(-1), expression.expr2))
        )


__all__ = ["CanonicalTransformer"]
