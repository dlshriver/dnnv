from __future__ import annotations

import numpy as np

from collections import defaultdict
from typing import List, Tuple

from .cnf import CnfTransformer
from .substitute_calls import SubstituteCalls
from ..expressions import *


def _extract_constants(expr: Add) -> Tuple[Add, Constant]:
    constants = []
    expressions = expr.expressions
    expr.expressions = []
    for e in expressions:
        if e.is_concrete:
            constants.append(-e)
        else:
            expr.expressions.append(e)
    return expr, Add(*constants).propagate_constants()


class CanonicalTransformer(CnfTransformer):
    def visit(self, expression: Expression) -> Expression:
        if self._top_level:
            expression = expression.propagate_constants()
            expression = SubstituteCalls().visit(expression)
        expr = super().visit(expression)
        return expr

    def visit_Add(self, expression: Add) -> Add:
        expressions = defaultdict(lambda: Constant(0))
        operands = [e for e in expression.expressions]
        while len(operands):
            expr = self.visit(operands.pop())
            if isinstance(expr, Add):
                if len(expr.expressions) > 1:
                    operands.extend(expr.expressions)
                    continue
                elif len(expr.expressions) == 1:
                    expr = expr.expressions[0]
                else:
                    continue
            symbol = expr
            value = Constant(1)
            if symbol.is_concrete:
                symbol, value = value, symbol.propagate_constants()
            elif isinstance(expr, Multiply):
                constants = []
                symbols = []
                for e in expr.expressions:
                    if e.is_concrete:
                        constants.append(e)
                    else:
                        symbols.append(e)
                if len(symbols) == 1:
                    symbol = symbols[0]
                else:
                    symbol = Multiply(*symbols)
                value = Multiply(*constants).propagate_constants()
            expressions[symbol] = expressions[symbol] + value
        products = []
        for v, c in expressions.items():
            const = c.propagate_constants()
            if np.all(const.value == 0):
                continue
            products.append(Multiply(const, v))
        return Add(*products)

    def visit_Equal(self, expression: Equal) -> And:
        expr1 = expression.expr1
        expr2 = expression.expr2
        return self.visit(And(expr1 <= expr2, expr2 <= expr1))

    def visit_GreaterThan(self, expression: GreaterThan) -> GreaterThan:
        lhs = self.visit_Add(
            Add(expression.expr1, Multiply(Constant(-1), expression.expr2))
        )
        lhs, rhs = _extract_constants(lhs)
        expr = GreaterThan(lhs, rhs)
        return expr

    def visit_GreaterThanOrEqual(
        self, expression: GreaterThanOrEqual
    ) -> GreaterThanOrEqual:
        lhs = self.visit_Add(
            Add(expression.expr1, Multiply(Constant(-1), expression.expr2))
        )
        lhs, rhs = _extract_constants(lhs)
        expr = GreaterThanOrEqual(lhs, rhs)
        return expr

    def visit_LessThan(self, expression: LessThan) -> GreaterThan:
        lhs = self.visit_Add(
            Add(Multiply(Constant(-1), expression.expr1), expression.expr2)
        )
        lhs, rhs = _extract_constants(lhs)
        expr = GreaterThan(lhs, rhs)
        return expr

    def visit_LessThanOrEqual(self, expression: LessThanOrEqual) -> GreaterThanOrEqual:
        lhs = self.visit_Add(
            Add(Multiply(Constant(-1), expression.expr1), expression.expr2)
        )
        lhs, rhs = _extract_constants(lhs)
        expr = GreaterThanOrEqual(lhs, rhs)
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
            const = Multiply(*consts).propagate_constants()
            product = Multiply(*symbols).propagate_constants()
            return Add(Multiply(const, product))
        return self.visit(Add(*[Multiply(*e) for e in expressions]))

    def visit_Negation(self, expression: Negation) -> Add:
        return self.visit_Multiply(Multiply(Constant(-1), expression.expr))

    def visit_NotEqual(self, expression: NotEqual) -> Or:
        expr1 = expression.expr1
        expr2 = expression.expr2
        return self.visit(Or(expr1 > expr2, expr2 > expr1))

    def visit_Subtract(self, expression: Subtract) -> Add:
        return self.visit_Add(
            Add(expression.expr1, Multiply(Constant(-1), expression.expr2))
        )


__all__ = ["CanonicalTransformer"]
