from ..expressions import (
    AssociativeExpression,
    BinaryExpression,
    Call,
    Constant,
    Expression,
    Image,
    Quantifier,
    Symbol,
    TernaryExpression,
    UnaryExpression,
)
from ..visitors import ExpressionVisitor


class ExpressionTransformer(ExpressionVisitor):
    def generic_visit(self, expression: Expression) -> Expression:
        if isinstance(expression, Expression):
            raise ValueError(
                f"Unimplemented expression type: {type(expression).__name__}"
            )
        return self.visit(Constant(expression))


class GenericExpressionTransformer(ExpressionTransformer):
    def generic_visit(self, expression: Expression) -> Expression:
        if isinstance(expression, AssociativeExpression):
            return self.visit_AssociativeExpression(expression)
        elif isinstance(expression, BinaryExpression):
            return self.visit_BinaryExpression(expression)
        elif isinstance(expression, Constant):
            return expression
        elif isinstance(expression, Call):
            return self.visit_Call(expression)
        elif isinstance(expression, Image):
            return self.visit_Image(expression)
        elif isinstance(expression, TernaryExpression):
            return self.visit_TernaryExpression(expression)
        elif isinstance(expression, Quantifier):
            variable = self.visit(expression.variable)
            expr = self.visit(expression.expression)
            return type(expression)(variable, expr)
        elif isinstance(expression, Symbol):
            return expression
        elif isinstance(expression, UnaryExpression):
            return self.visit_UnaryExpression(expression)
        return super().generic_visit(expression)

    def visit_AssociativeExpression(
        self, expression: AssociativeExpression
    ) -> Expression:
        exprs = [self.visit(expr) for expr in expression.expressions]
        return type(expression)(*exprs)

    def visit_BinaryExpression(self, expression: BinaryExpression) -> Expression:
        expr1 = self.visit(expression.expr1)
        expr2 = self.visit(expression.expr2)
        return type(expression)(expr1, expr2)

    def visit_Call(self, expression: Call) -> Expression:
        function = self.visit(expression.function)
        args = tuple(self.visit(arg) for arg in expression.args)
        kwargs = {k: self.visit(v) for k, v in expression.kwargs.items()}
        return Call(function, args, kwargs)

    def visit_Image(self, expression: Image) -> Expression:
        path = expression.path
        if isinstance(expression.path, Expression):
            path = self.visit(expression.path)
        return type(expression)(path)

    def visit_TernaryExpression(self, expression: TernaryExpression) -> Expression:
        expr1 = self.visit(expression.expr1)
        expr2 = self.visit(expression.expr2)
        expr3 = self.visit(expression.expr3)
        return type(expression)(expr1, expr2, expr3)

    def visit_UnaryExpression(self, expression: UnaryExpression) -> Expression:
        expr = self.visit(expression.expr)
        return type(expression)(expr)


__all__ = ["ExpressionTransformer", "GenericExpressionTransformer"]
