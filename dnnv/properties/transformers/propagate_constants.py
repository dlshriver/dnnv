from __future__ import annotations

from typing import Union

import numpy as np

from dnnv.nn.graph import OperationGraph

from ..expressions import (
    ArithmeticExpression,
    AssociativeExpression,
    BinaryExpression,
    Call,
    Constant,
    Divide,
    Exists,
    Expression,
    Forall,
    IfThenElse,
    Image,
    Implies,
    LogicalExpression,
    Network,
    Parameter,
    Subscript,
    Subtract,
    Symbol,
    TernaryExpression,
    UnaryExpression,
)
from .base import GenericExpressionTransformer


def _is_concrete(expression: Expression) -> bool:
    # cheap concreteness check for propagating constants
    return isinstance(expression, Constant) or (
        isinstance(expression, Network) and expression.is_concrete
    )


class PropagateConstants(GenericExpressionTransformer):
    def visit_AssociativeExpression(
        self, expression: AssociativeExpression
    ) -> Expression:
        expression_type = type(expression)
        symbolic_expressions = []
        concrete_expressions = []
        for expr in expression.expressions:
            expr = self.visit(expr)
            if not _is_concrete(expr):
                symbolic_expressions.append(expr)
            else:
                expr_value = expr.value
                for dominant_value in expression.DOMINANT_VALUES:
                    if np.all(expr_value == dominant_value):
                        return Constant(expr_value)
                concrete_expressions.append(expr)
        if len(symbolic_expressions) == 0:
            new_expr = expression_type(*concrete_expressions, ctx=expression.ctx)
            return Constant(new_expr.value)
        if len(symbolic_expressions) == 1 and len(concrete_expressions) == 0:
            return symbolic_expressions[0]
        if len(concrete_expressions) == 0:
            return expression_type(*symbolic_expressions, ctx=expression.ctx)
        if len(concrete_expressions) == 1:
            concrete_expr_value = concrete_expressions[0].value
        elif len(concrete_expressions) > 1:
            concrete_expr_value = expression_type(
                *concrete_expressions, ctx=expression.ctx
            ).value
        if np.all(concrete_expr_value == expression.BASE_VALUE):
            if len(symbolic_expressions) == 1:
                return symbolic_expressions[0]
            new_expr = expression_type(*symbolic_expressions, ctx=expression.ctx)
        else:
            new_expr = expression_type(
                Constant(concrete_expr_value),
                *symbolic_expressions,
                ctx=expression.ctx,
            )
        return new_expr

    def visit_BinaryExpression(
        self, expression: BinaryExpression
    ) -> Union[BinaryExpression, Constant]:
        expr1 = self.visit(expression.expr1)
        expr2 = self.visit(expression.expr2)
        new_expr = type(expression)(expr1, expr2, ctx=expression.ctx)
        if not _is_concrete(expr1) or not _is_concrete(expr2):
            return new_expr
        return Constant(new_expr.value)

    def visit_Call(self, expression: Call) -> Expression:
        function = self.visit(expression.function)
        args = tuple(self.visit(arg) for arg in expression.args)
        kwargs = {k: self.visit(v) for k, v in expression.kwargs.items()}
        if (
            _is_concrete(function)
            and all(_is_concrete(a) for a in args)
            and all(_is_concrete(a) for a in kwargs.values())
        ):
            result = expression.value
            if isinstance(result, Expression):
                return result.propagate_constants()
            if isinstance(result, OperationGraph):
                return Network(str(expression)).concretize(result)
            return Constant(result)
        return Call(function, args, kwargs)

    def visit_Constant(self, expression: Constant) -> Constant:
        return expression

    def visit_Divide(self, expression: Divide) -> Expression:
        expr1 = self.visit(expression.expr1)
        expr2 = self.visit(expression.expr2)
        assert isinstance(expr1, ArithmeticExpression)
        assert isinstance(expr2, ArithmeticExpression)
        if isinstance(expr1, Constant) and isinstance(expr2, Constant):
            return Constant(expr1.value / expr2.value)
        elif expr1.is_equivalent(expr2):
            return Constant(1)
        elif isinstance(expr2, Constant) and isinstance(expr2.value, (float, int)):
            if expr2.value == 1:
                return expr1
            elif expr2.value == -1:
                return -expr1
            elif expr2.value == 0:
                raise ZeroDivisionError(expression)
        elif (
            isinstance(expr1, Constant)
            and isinstance(expr1.value, (float, int))
            and expr1.value == 0
        ):
            return Constant(expr1.value)
        return Divide(expr1, expr2)

    def visit_Exists(self, expression: Exists) -> Union[Constant, Exists]:
        variable = self.visit(expression.variable)
        expr = self.visit(expression.expression)
        if isinstance(expr, Constant):
            return Constant(bool(expr.value))
        assert isinstance(variable, Symbol)
        return Exists(variable, expr)

    def visit_Forall(self, expression: Forall) -> Expression:
        variable = self.visit(expression.variable)
        expr = self.visit(expression.expression)
        if isinstance(expr, Constant):
            return Constant(bool(expr.value))
        elif variable not in expr.variables:
            return expr
        assert isinstance(variable, Symbol)
        return Forall(variable, expr)

    def visit_IfThenElse(self, expression: IfThenElse) -> Expression:
        condition = self.visit(expression.condition)
        t_expr = self.visit(expression.t_expr)
        f_expr = self.visit(expression.f_expr)
        bool_type = (bool, np.bool_)
        if _is_concrete(condition):
            if condition.value:
                return t_expr
            return f_expr
        elif _is_concrete(t_expr) and _is_concrete(f_expr):
            assert isinstance(condition, LogicalExpression)
            if t_expr.is_equivalent(f_expr):
                return Constant(t_expr)
            elif isinstance(t_expr.value, bool_type) and isinstance(
                f_expr.value, bool_type
            ):
                if t_expr.value:
                    return condition
                return ~condition
        return IfThenElse(condition, t_expr, f_expr)

    def visit_Image(self, expression: Image) -> Image:
        path = expression.path
        if isinstance(path, Expression):
            path = self.visit(path)
        return Image.load(path)

    def visit_Implies(self, expression: Implies) -> Expression:
        antecedent = self.visit(expression.expr1)
        consequent = self.visit(expression.expr2)
        if antecedent.is_equivalent(consequent):
            return Constant(True)
        if _is_concrete(antecedent) and _is_concrete(consequent):
            antecedent_value = antecedent.value
            consequent_value = consequent.value
            if np.all(consequent_value):
                return Constant(True)
            elif isinstance(antecedent_value, np.ndarray):
                if np.all(antecedent_value):
                    if np.all(~consequent_value):
                        return Constant(False)
                    return Constant(consequent_value)
                if np.all(~antecedent_value):
                    return Constant(~antecedent_value)
                if isinstance(consequent_value, np.ndarray):
                    return Constant(~antecedent_value | consequent_value)
            elif antecedent_value:
                return Constant(consequent_value)
            else:
                return Constant(True)
        elif _is_concrete(antecedent):
            antecedent_value = antecedent.value
            if isinstance(antecedent_value, np.ndarray):
                if np.all(antecedent_value):
                    return consequent
                if np.all(~antecedent_value):
                    return Constant(True)
            elif antecedent_value:
                return consequent
            else:
                return Constant(True)
        elif _is_concrete(consequent):
            assert isinstance(antecedent, LogicalExpression)
            consequent_value = consequent.value
            if isinstance(consequent_value, np.ndarray):
                if np.all(consequent_value):
                    return Constant(True)
                if np.all(~consequent_value):
                    return ~antecedent
            elif consequent_value:
                return Constant(True)
            else:
                return ~antecedent
        return Implies(antecedent, consequent)

    def visit_Network(self, expression: Network) -> Union[Constant, Network]:
        return expression

    def visit_Parameter(self, expression: Parameter) -> Union[Constant, Parameter]:
        if expression.is_concrete:
            return Constant(expression.value)
        return expression

    def visit_Subtract(self, expression: Subtract) -> Expression:
        expr1 = self.visit(expression.expr1)
        expr2 = self.visit(expression.expr2)
        assert isinstance(expr1, ArithmeticExpression)
        assert isinstance(expr2, ArithmeticExpression)
        if expr1.is_equivalent(expr2):
            return Constant(0)
        if isinstance(expr1, Constant) and isinstance(expr2, Constant):
            return Constant(expr1.value - expr2.value)
        if isinstance(expr1, Constant) and np.all(expr1.value == 0):
            return -expr2
        if isinstance(expr2, Constant) and np.all(expr2.value == 0):
            return expr1
        return Subtract(expr1, expr2)

    def visit_Subscript(self, expression: Subscript) -> Expression:
        expr1 = self.visit(expression.expr1)
        expr2 = self.visit(expression.expr2)
        if isinstance(expr1, Network) and expr1.is_concrete and expr2.is_concrete:
            return expr1[expr2.value]
        new_expr = type(expression)(expr1, expr2, ctx=expression.ctx)
        if not _is_concrete(expr1) or not _is_concrete(expr2):
            return new_expr
        return Constant(new_expr.value)

    def visit_Symbol(self, expression: Symbol) -> Union[Constant, Symbol]:
        if expression.is_concrete:
            return Constant(expression.value)
        return expression

    def visit_TernaryExpression(
        self, expression: TernaryExpression
    ) -> Union[TernaryExpression, Constant]:
        expr1 = self.visit(expression.expr1)
        expr2 = self.visit(expression.expr2)
        expr3 = self.visit(expression.expr3)
        concrete = True
        if (
            not _is_concrete(expr1)
            or not _is_concrete(expr2)
            or not _is_concrete(expr3)
        ):
            concrete = False
        new_expr = type(expression)(expr1, expr2, expr3, ctx=expression.ctx)
        if concrete:
            return Constant(new_expr.value)
        return new_expr

    def visit_UnaryExpression(
        self, expression: UnaryExpression
    ) -> Union[UnaryExpression, Constant]:
        expr = self.visit(expression.expr)
        concrete = True
        if not _is_concrete(expr):
            concrete = False
        new_expr = type(expression)(expr, ctx=expression.ctx)
        if concrete:
            return Constant(new_expr.value)
        return new_expr


__all__ = ["PropagateConstants"]
