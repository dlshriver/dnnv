from __future__ import annotations

from dnnv.properties.expressions.base import Expression

from ...expressions import BinaryExpression, Call
from ..base import GenericExpressionTransformer
from ._calls import FunctionSubstitutor


class SubstituteCalls(GenericExpressionTransformer):
    def visit_BinaryExpression(self, expression: BinaryExpression) -> BinaryExpression:
        expr_type = type(expression)
        expr1 = expression.expr1
        expr2 = expression.expr2
        if isinstance(expr1, Call) and expr1.function.is_concrete:
            substitutor = FunctionSubstitutor.lookup(expr1.function.value)
            binexpr_substitute_method = f"substitute_{expr_type.__name__}"
            if substitutor is not None and hasattr(
                substitutor, binexpr_substitute_method
            ):
                result = getattr(substitutor, binexpr_substitute_method)(expr1, expr2)
                if result is not NotImplemented:
                    return self.visit(result)
        elif isinstance(expr2, Call) and expr2.function.is_concrete:
            substitutor = FunctionSubstitutor.lookup(expr2.function.value)
            binexpr_substitute_method = f"substitute_{expr_type.__name__}"
            if substitutor is not None and hasattr(
                substitutor, binexpr_substitute_method
            ):
                result = getattr(substitutor, binexpr_substitute_method)(expr1, expr2)
                if result is not NotImplemented:
                    return self.visit(result)
        return expr_type(self.visit(expr1), self.visit(expr2))

    def visit_Call(self, expression: Call) -> Expression:
        function = self.visit(expression.function)
        args = tuple([self.visit(arg) for arg in expression.args])
        kwargs = {name: self.visit(value) for name, value in expression.kwargs.items()}
        if function.is_concrete:
            substitutor = FunctionSubstitutor.lookup(function.value)
            if substitutor is not None:
                result = substitutor(function, *args, **kwargs)
                if result is not NotImplemented:
                    return result
        expr = Call(function, args, kwargs)
        return expr


__all__ = ["SubstituteCalls"]
