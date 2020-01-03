from typing import List, Optional, Tuple

from .base import *
from .visitors import ExpressionVisitor


class ExpressionTransformer(ExpressionVisitor):
    def __init__(self):
        self.visited = {}

    def visit(self, expression):
        expression_id = id(expression)
        if expression_id not in self.visited:
            method_name = "visit_%s" % expression.__class__.__name__
            visitor = getattr(self, method_name, self.generic_visit)
            self.visited[expression_id] = visitor(expression)
        return self.visited[expression_id]

    def generic_visit(self, expression):
        for name, value in expression.__dict__.items():
            if isinstance(value, Expression):
                new_value = self.visit(value)
                setattr(expression, name, new_value)
            elif isinstance(value, (list, tuple, set)):
                new_value = []
                for sub_value in value:
                    if isinstance(sub_value, Expression):
                        new_sub_value = self.visit(sub_value)
                        if new_sub_value is not None:
                            new_value.append(new_sub_value)
                    else:
                        new_value.append(sub_value)
                setattr(expression, name, new_value)
        return expression


class PropagateConstants(ExpressionTransformer):
    def generic_visit(self, expression):
        if isinstance(expression, Expression):
            raise ValueError(
                f"Unimplemented expression type: {type(expression).__name__}"
            )
        return Constant(expression)

    def visit_Attribute(self, expression: Attribute):
        expr = self.visit(expression.expr)
        name = self.visit(expression.name)
        if isinstance(expr, Constant) and isinstance(name, Constant):
            return Constant(getattr(expr.value, name.value))
        return Attribute(expr, name)

    def visit_Add(self, expression: Add):
        expressions = []
        constant_expressions = []
        for expr in expression.expressions:
            expr = self.visit(expr)
            if isinstance(expr, Constant):
                constant_expressions.append(expr)
            else:
                expressions.append(expr)
        if len(constant_expressions) == 0 and len(expressions) == 0:
            return Constant(0)
        elif len(constant_expressions) == 0:
            return Add(*expressions)
        constant_value = 0
        for expr in constant_expressions:
            constant_value += expr.value
        if len(expressions) == 0:
            return Constant(constant_value)
        return Add(Constant(constant_value), *expressions)

    def visit_And(self, expression: And):
        expressions = []
        constant_expressions = []
        for expr in expression.expressions:
            expr = self.visit(expr)
            if isinstance(expr, Constant):
                constant_expressions.append(expr)
            else:
                expressions.append(expr)
        if len(constant_expressions) == 0 and len(expressions) == 0:
            return Constant(True)
        constant_value = True
        for expr in constant_expressions:
            constant_value &= expr.value
        if not constant_value or len(expressions) == 0:
            return Constant(constant_value)
        if len(expressions) == 1:
            return expressions[0]
        return And(*expressions)

    def visit_Constant(self, expression: Constant):
        return expression

    def visit_Divide(self, expression: Divide):
        expr1 = self.visit(expression.expr1)
        expr2 = self.visit(expression.expr2)
        if isinstance(expr1, Constant) and isinstance(expr2, Constant):
            return Constant(expr1.value / expr2.value)
        return Divide(expr1, expr2)

    def visit_Equal(self, expression: Equal):
        expr1 = self.visit(expression.expr1)
        expr2 = self.visit(expression.expr2)
        if isinstance(expr1, Constant) and isinstance(expr2, Constant):
            return Constant(expr1.value == expr2.value)
        return Equal(expr1, expr2)

    def visit_Forall(self, expression: Forall):
        variable = self.visit(expression.variable)
        expr = self.visit(expression.expression)
        if isinstance(expr, Constant):
            return Constant(bool(expr))
        return Forall(variable, expr)

    def visit_FunctionCall(self, expression: FunctionCall):
        function = self.visit(expression.function)
        args = tuple([self.visit(arg) for arg in expression.args])
        kwargs = {name: self.visit(value) for name, value in expression.kwargs.items()}
        params = args + tuple(kwargs.values())
        if (isinstance(function, Constant) or function.is_concrete) and all(
            isinstance(param, Constant) for param in params
        ):
            args = tuple(arg.value for arg in args)
            kwargs = {name: value.value for name, value in kwargs.items()}
            result = function.value(*args, **kwargs)
            return Constant(result)
        return FunctionCall(function, args, kwargs)

    def visit_GreaterThan(self, expression: GreaterThan):
        expr1 = self.visit(expression.expr1)
        expr2 = self.visit(expression.expr2)
        if isinstance(expr1, Constant) and isinstance(expr2, Constant):
            return Constant(expr1.value > expr2.value)
        return GreaterThan(expr1, expr2)

    def visit_GreaterThanOrEqual(self, expression: GreaterThanOrEqual):
        expr1 = self.visit(expression.expr1)
        expr2 = self.visit(expression.expr2)
        if isinstance(expr1, Constant) and isinstance(expr2, Constant):
            return Constant(expr1.value >= expr2.value)
        return GreaterThanOrEqual(expr1, expr2)

    def visit_Image(self, expression: Image):
        path = self.visit(expression.path)
        return Image.load(path)

    def visit_Implies(self, expression: Implies):
        antecedent = self.visit(expression.expr1)
        consequent = self.visit(expression.expr2)
        if isinstance(antecedent, Constant) and antecedent.value == False:
            return Constant(True)
        elif isinstance(antecedent, Constant) and antecedent.value == True:
            return consequent
        elif isinstance(antecedent, Constant) and isinstance(consequent, Constant):
            return Constant(~(antecedent.value) | consequent.value)
        return Implies(antecedent, consequent)

    def visit_LessThan(self, expression: LessThan):
        expr1 = self.visit(expression.expr1)
        expr2 = self.visit(expression.expr2)
        if isinstance(expr1, Constant) and isinstance(expr2, Constant):
            return Constant(expr1.value < expr2.value)
        return LessThan(expr1, expr2)

    def visit_LessThanOrEqual(self, expression: LessThanOrEqual):
        expr1 = self.visit(expression.expr1)
        expr2 = self.visit(expression.expr2)
        if isinstance(expr1, Constant) and isinstance(expr2, Constant):
            return Constant(expr1.value <= expr2.value)
        return LessThanOrEqual(expr1, expr2)

    def visit_Multiply(self, expression: Multiply):
        expressions = []
        constant_expressions = []
        for expr in expression.expressions:
            expr = self.visit(expr)
            if isinstance(expr, Constant):
                constant_expressions.append(expr)
            else:
                expressions.append(expr)
        if len(constant_expressions) == 0 and len(expressions) == 0:
            return Constant(1)
        constant_value = 1
        for expr in constant_expressions:
            constant_value *= expr.value
        if constant_value == 0:
            return Constant(0)
        elif len(expressions) == 0:
            return Constant(constant_value)
        elif constant_value == 1:
            if len(expressions) > 1:
                return Multiply(*expressions)
            else:
                return expressions[0]
        return Multiply(Constant(constant_value), *expressions)

    def visit_Network(self, expression: Network):
        return expression

    def visit_NotEqual(self, expression: NotEqual):
        expr1 = self.visit(expression.expr1)
        expr2 = self.visit(expression.expr2)
        if isinstance(expr1, Constant) and isinstance(expr2, Constant):
            return Constant(expr1.value != expr2.value)
        return NotEqual(expr1, expr2)

    def visit_Or(self, expression: Or):
        expressions = []
        constant_expressions = []
        for expr in expression.expressions:
            expr = self.visit(expr)
            if isinstance(expr, Constant):
                constant_expressions.append(expr)
            else:
                expressions.append(expr)
        if len(constant_expressions) == 0 and len(expressions) == 0:
            return Constant(False)
        constant_value = False
        for expr in constant_expressions:
            constant_value |= expr.value
        if constant_value or len(expressions) == 0:
            return Constant(constant_value)
        if len(expressions) == 1:
            return expressions[0]
        return Or(*expressions)

    def visit_Parameter(self, expression: Parameter):
        if expression.is_concrete:
            return Constant(expression.value)
        return expression

    def visit_Subscript(self, expression: Subscript):
        expr = self.visit(expression.expr)
        index = self.visit(expression.index)
        if isinstance(expr, Constant) and isinstance(index, Constant):
            return Constant(expr.value[index.value])
        return Subscript(expr, index)

    def visit_Subtract(self, expression: Subtract):
        expr1 = self.visit(expression.expr1)
        expr2 = self.visit(expression.expr2)
        if isinstance(expr1, Constant) and isinstance(expr2, Constant):
            return Constant(expr1.value - expr2.value)
        return Subtract(expr1, expr2)

    def visit_Symbol(self, expression: Symbol):
        if expression.is_concrete:
            return Constant(expression.value)
        return expression


class ToCNF(ExpressionTransformer):
    def visit_And(self, expression: And) -> And:
        expressions = []  # type: List[Expression]
        for expr in expression.expressions:
            expr = self.visit(expr)
            if isinstance(expr, And):
                expressions.extend(expr.expressions)
            else:
                expressions.append(expr)
        if len(expressions) == 1:
            return And(expressions[0])
        return And(*expressions)

    def visit_Forall(self, expression: Forall):
        expr = self.visit(expression.expression)
        return expr

    def visit_Implies(self, expression: Implies) -> And:
        return self.visit(Or(~expression.expr1, expression.expr2))

    def visit_Or(self, expression: Or) -> And:
        expressions = []  # type: List[Expression]
        conjunction = None  # type: Optional[And]
        for expr in expression.expressions:
            expr = self.visit(expr)
            if isinstance(expr, Or):
                expressions.extend(expr.expressions)
            elif isinstance(expr, And) and conjunction is None:
                conjunction = expr
            else:
                expressions.append(expr)
        if conjunction is None:
            if len(expressions) == 0:
                return And(Constant(True))
            elif len(expressions) == 1:
                return And(expressions[0])
            return And(Or(*expressions))
        if len(expressions) == 0:
            return conjunction
        clauses = list(
            (e,) for e in conjunction.expressions
        )  # type: List[Tuple[Expression, ...]]
        for expr in expressions:
            if isinstance(expr, And):
                new_clauses = []
                for clause in clauses:
                    for e in expr.expressions:
                        new_clauses.append(clause + (e,))
                clauses = new_clauses
            else:
                assert not isinstance(expr, Or)
                new_clauses = []
                for clause in clauses:
                    new_clauses.append(clause + (expr,))
                clauses = new_clauses
        return And(*[Or(*clause) for clause in clauses])
