from __future__ import annotations

from dnnv.properties.expressions.base import Expression

from ...expressions import BinaryExpression, Call
from ...visitors import DetailsInference
from ..base import GenericExpressionTransformer
from ._calls import FunctionSubstitutor


class SubstituteCalls(GenericExpressionTransformer):
    def __init__(self, form="dnf"):
        super().__init__()
        # `form` provides a hint to the substitutor on how to efficiently
        # format the substitution expression
        self.form = form
        self._infer_expression_details = lambda: None

    def _lazy_inference(self):
        self._infer_expression_details()
        self._infer_expression_details = lambda: None

    def visit(self, expression):
        if self._top_level:
            self._infer_expression_details = lambda: DetailsInference().visit(
                expression
            )
        return super().visit(expression)

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
                self._lazy_inference()
                result = getattr(substitutor, binexpr_substitute_method)(
                    expr1, expr2, form=self.form
                )
                if result is not NotImplemented:
                    return self.visit(result)
        elif isinstance(expr2, Call) and expr2.function.is_concrete:
            substitutor = FunctionSubstitutor.lookup(expr2.function.value)
            binexpr_substitute_method = f"substitute_{expr_type.__name__}"
            if substitutor is not None and hasattr(
                substitutor, binexpr_substitute_method
            ):
                self._lazy_inference()
                result = getattr(substitutor, binexpr_substitute_method)(
                    expr1, expr2, form=self.form
                )
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
                self._lazy_inference()
                result = substitutor(function, *args, **kwargs)
                if result is not NotImplemented:
                    return result
        expr = Call(function, args, kwargs)
        return expr

    def visit_Not(self, expression):
        form = self.form
        self.form = "cnf" if form == "dnf" else "dnf"
        result = super().generic_visit(expression)
        self.form = form
        return result


__all__ = ["SubstituteCalls"]
