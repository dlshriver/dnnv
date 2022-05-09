from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Tuple, Union, cast

import numpy as np

from ..expressions import *
from .dnf import DnfTransformer


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
    def _extract_coefficients(
        self, expression: Add
    ) -> Dict[Expression, ArithmeticExpression]:
        coefficients: Dict[Expression, ArithmeticExpression] = defaultdict(
            lambda: Constant(0)
        )
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

    def visit_Equal(self, expression: Equal) -> Union[Constant, Or]:
        expr1 = expression.expr1
        expr2 = expression.expr2
        expr = self.visit(Or(And(expr1 <= expr2, expr2 <= expr1)))
        assert isinstance(expr, (Constant, Or))
        if expr.is_concrete:
            return Constant(expr.value)
        return expr

    def visit_GreaterThan(self, expression: GreaterThan) -> Union[Constant, Or]:
        lhs, rhs = _extract_constants(
            cast(
                Add,
                self.visit(
                    Add(
                        Multiply(Constant(-1), expression.expr1),
                        expression.expr2,
                    )
                ),
            )
        )
        expr = Or(And(LessThan(lhs, rhs)))
        if expr.is_concrete:
            return Constant(expr.value)
        return expr

    def visit_GreaterThanOrEqual(
        self, expression: GreaterThanOrEqual
    ) -> Union[Constant, Or]:
        lhs, rhs = _extract_constants(
            cast(
                Add,
                self.visit(
                    Add(
                        Multiply(Constant(-1), expression.expr1),
                        expression.expr2,
                    )
                ),
            )
        )
        expr = Or(And(LessThanOrEqual(lhs, rhs)))
        if expr.is_concrete:
            return Constant(expr.value)
        return expr

    def visit_LessThan(self, expression: LessThan) -> Union[Constant, Or]:
        lhs, rhs = _extract_constants(
            cast(
                Add,
                self.visit(
                    Add(
                        expression.expr1,
                        Multiply(Constant(-1), expression.expr2),
                    )
                ),
            )
        )
        expr = Or(And(LessThan(lhs, rhs)))
        if expr.is_concrete:
            return Constant(expr.value)
        return expr

    def visit_LessThanOrEqual(self, expression: LessThanOrEqual) -> Union[Constant, Or]:
        lhs, rhs = _extract_constants(
            cast(
                Add,
                self.visit(
                    Add(
                        expression.expr1,
                        Multiply(Constant(-1), expression.expr2),
                    )
                ),
            )
        )
        expr = Or(And(LessThanOrEqual(lhs, rhs)))
        if expr.is_concrete:
            return Constant(expr.value)
        return expr

    def visit_Multiply(self, expression: Multiply) -> Add:
        expression_lists: List[List[Expression]] = [[]]
        for expr in expression.expressions:
            expr = self.visit(expr)
            if isinstance(expr, Add):
                new_expression_lists = []
                for elist in expression_lists:
                    for e in expr.expressions:
                        new_elist = list(elist)
                        new_elist.append(e)
                        new_expression_lists.append(new_elist)
                expression_lists = new_expression_lists
            else:
                for elist in expression_lists:
                    elist.append(expr)
        if len(expression_lists) == 0:
            return Constant(0)
        if len(expression_lists) <= 1:
            consts: List[Expression] = []
            symbols: List[Expression] = [Constant(1)]
            for e in expression_lists[0]:
                if e.is_concrete:
                    consts.append(e)
                else:
                    symbols.append(e)
            const = Constant(Multiply(*consts).value)
            product = Multiply(*symbols).propagate_constants()
            return Add(Multiply(const, product))
        return self.visit(Add(*[Multiply(*e) for e in expression_lists]))

    def visit_Negation(self, expression: Negation) -> Add:
        return self.visit(Multiply(Constant(-1), expression.expr))

    def visit_NotEqual(self, expression: NotEqual) -> Union[Constant, Or]:
        expr1 = expression.expr1
        expr2 = expression.expr2
        expr = self.visit(Or(And(expr1 < expr2), And(expr2 < expr1)))
        assert isinstance(expr, (Constant, Or))
        if expr.is_concrete:
            return Constant(expr.value)
        return expr

    def visit_Subtract(self, expression: Subtract) -> Add:
        return self.visit(
            Add(expression.expr1, Multiply(Constant(-1), expression.expr2))
        )


__all__ = ["CanonicalTransformer"]
