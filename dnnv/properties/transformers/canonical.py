from __future__ import annotations

import numpy as np
import operator

from collections import defaultdict
from functools import reduce
from typing import Dict, List, Tuple, Union

from .cnf import CnfTransformer
from .fol import FirstOrderLogicTransformer
from .substitute_calls import SubstituteCalls
from ..expressions import *


def _extract_constants(expr: Add) -> Tuple[Add, Constant]:
    constants = []
    expressions = []
    for e in expr.expressions:
        if e.is_concrete:
            constants.append(-e.value)
        else:
            expressions.append(e)
    return Add(*expressions), Constant(sum(constants))


class CanonicalTransformer(CnfTransformer):
    def visit(self, expression: Expression) -> Expression:
        if self._top_level:
            expression = FirstOrderLogicTransformer().visit(expression)
            expression = expression.propagate_constants()
            expression = SubstituteCalls().visit(expression)
        expr = super().visit(expression)
        return expr

    def visit_Add(self, expression: Add) -> Add:
        expressions: Dict[Expression, Union[int, float]] = defaultdict(lambda: 0)
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
            value = 1
            if symbol.is_concrete:
                symbol, value = Constant(value), symbol.value
            elif isinstance(expr, Multiply):
                constants = []
                symbols = []
                for e in expr.expressions:
                    if e.is_concrete:
                        constants.append(e.value)
                    else:
                        symbols.append(e)
                if len(symbols) == 1:
                    symbol = symbols[0]
                else:
                    symbol = Multiply(*symbols)
                value = reduce(operator.mul, constants, 1)
            expressions[symbol] = expressions[symbol] + value
        products = []
        for v, c in expressions.items():
            if np.all(c == 0):
                continue
            products.append(Multiply(Constant(c), v))
        return Add(*products)

    def visit_Equal(self, expression: Equal) -> And:
        expr1 = expression.expr1
        expr2 = expression.expr2
        return self.visit_And(And(expr1 <= expr2, expr2 <= expr1))

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
            symbols: List[Expression] = [Constant(1)]
            for expr in expressions[0]:
                if expr.is_concrete:
                    consts.append(expr.value)
                else:
                    symbols.append(expr)
            const = Constant(reduce(operator.mul, consts, 1))
            product = Multiply(*symbols).propagate_constants()
            return Add(Multiply(const, product))
        return self.visit_Add(Add(*[Multiply(*e) for e in expressions]))

    def visit_Negation(self, expression: Negation) -> Add:
        return self.visit_Multiply(Multiply(Constant(-1), expression.expr))

    def visit_NotEqual(self, expression: NotEqual) -> And:
        expr1 = expression.expr1
        expr2 = expression.expr2
        return self.visit_Or(Or(expr1 > expr2, expr2 > expr1))

    def visit_Subtract(self, expression: Subtract) -> Add:
        return self.visit_Add(
            Add(expression.expr1, Multiply(Constant(-1), expression.expr2))
        )


__all__ = ["CanonicalTransformer"]
