from __future__ import annotations

from ..expressions import Expression
from .base import GenericExpressionTransformer


class SubstituteExpression(GenericExpressionTransformer):
    def __init__(self, from_expr, to_expr):
        super().__init__()
        self.from_expr = from_expr
        self.to_expr = to_expr

    def visit(self, expression: Expression) -> Expression:
        if expression.is_equivalent(self.from_expr):
            return self.to_expr
        return super().generic_visit(expression)


__all__ = ["SubstituteExpression"]
