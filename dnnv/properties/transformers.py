from .base import *


class ExpressionTransformer(ExpressionVisitor):
    def visit(self, expression):
        method_name = "visit_%s" % expression.__class__.__name__
        visitor = getattr(self, method_name, self.generic_visit)
        return visitor(expression)

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
                "Unimplemented expression type: %s" % type(expression).__name__
            )
        return Constant(expression)

    def visit_Add(self, expression: "Add"):
        expr1 = self.visit(expression.expr1)
        expr2 = self.visit(expression.expr2)
        if isinstance(expr1, Constant) and isinstance(expr2, Constant):
            return expr1 + expr2
        return Add(expr1, expr2)

    def visit_And(self, expression: "And"):
        expressions = []
        constant_expressions = []
        for expr in expression.expressions:
            expr = self.visit(expr)
            if isinstance(expr, Constant):
                constant_expressions.append(expr)
            else:
                expressions.append(expr)
        constant_value = Constant(True)
        for expr in constant_expressions:
            constant_value &= expr
        if constant_value.value and len(expressions) > 0:
            if len(expressions) == 1:
                return expressions[0]
            return And(*expressions)
        return constant_value

    def visit_Constant(self, expression: "Constant"):
        return expression

    def visit_Divide(self, expression: "Divide"):
        expr1 = self.visit(expression.expr1)
        expr2 = self.visit(expression.expr2)
        if isinstance(expr1, Constant) and isinstance(expr2, Constant):
            return expr1 / expr2
        return Divide(expr1, expr2)

    def visit_Equal(self, expression: "Equal"):
        expr1 = self.visit(expression.expr1)
        expr2 = self.visit(expression.expr2)
        if isinstance(expr1, Constant) and isinstance(expr2, Constant):
            return expr1 == expr2
        return Equal(expr1, expr2)

    def visit_Forall(self, expression: "Forall"):
        variable = self.visit(expression.variable)
        expr = self.visit(expression.expression)
        if isinstance(expr, Constant):
            return Constant(bool(expr))
        return Forall(variable, expr)

    def visit_Function(self, expression: "Function"):
        return super().generic_visit(expression)

    def visit_FunctionCall(self, expression: "FunctionCall"):
        function = self.visit(expression.function)
        args = tuple([self.visit(arg) for arg in expression.args])
        kwargs = {name: self.visit(value) for name, value in expression.kwargs.items()}
        params = args + tuple(kwargs.values())
        if function.concrete_value is not None and all(
            isinstance(param, Constant) for param in params
        ):
            args = tuple(arg.value for arg in args)
            kwargs = {name: value.value for name, value in kwargs.items()}
            result = function.concrete_value(*args, **kwargs)
            return Constant(result)
        return FunctionCall(
            function, args, kwargs, is_network_output=expression.is_network_output
        )

    def visit_GreaterThan(self, expression: "GreaterThan"):
        expr1 = self.visit(expression.expr1)
        expr2 = self.visit(expression.expr2)
        if isinstance(expr1, Constant) and isinstance(expr2, Constant):
            return expr1 > expr2
        return GreaterThan(expr1, expr2)

    def visit_GreaterThanOrEqual(self, expression: "GreaterThanOrEqual"):
        expr1 = self.visit(expression.expr1)
        expr2 = self.visit(expression.expr2)
        if isinstance(expr1, Constant) and isinstance(expr2, Constant):
            return expr1 >= expr2
        return GreaterThanOrEqual(expr1, expr2)

    def visit_Image(self, expression: "Image"):
        return expression

    def visit_Implies(self, expression: "Implies"):
        antecedent = self.visit(expression.antecedent)
        consequent = self.visit(expression.consequent)
        if isinstance(antecedent, Constant) and antecedent.value == False:
            return Constant(True)
        elif isinstance(antecedent, Constant) and antecedent.value == True:
            return consequent
        elif isinstance(antecedent, Constant) and isinstance(consequent, Constant):
            return ~antecedent | consequent
        return Implies(antecedent, consequent)

    def visit_LessThan(self, expression: "LessThan"):
        expr1 = self.visit(expression.expr1)
        expr2 = self.visit(expression.expr2)
        if isinstance(expr1, Constant) and isinstance(expr2, Constant):
            return expr1 < expr2
        return LessThan(expr1, expr2)

    def visit_LessThanOrEqual(self, expression: "LessThanOrEqual"):
        expr1 = self.visit(expression.expr1)
        expr2 = self.visit(expression.expr2)
        if isinstance(expr1, Constant) and isinstance(expr2, Constant):
            return expr1 <= expr2
        return LessThanOrEqual(expr1, expr2)

    def visit_Network(self, expression: "Network"):
        return expression

    def visit_NotEqual(self, expression: "NotEqual"):
        expr1 = self.visit(expression.expr1)
        expr2 = self.visit(expression.expr2)
        if isinstance(expr1, Constant) and isinstance(expr2, Constant):
            return expr1 != expr2
        return NotEqual(expr1, expr2)

    def visit_Or(self, expression: "Or"):
        expressions = []
        constant_expressions = []
        for expr in expression.expressions:
            expr = self.visit(expr)
            if isinstance(expr, Constant):
                constant_expressions.append(expr)
            else:
                expressions.append(expr)
        constant_value = Constant(False)
        for expr in constant_expressions:
            constant_value |= expr
        if constant_value.value or len(expressions) == 0:
            return constant_value
        elif len(expressions) == 1:
            return expressions[0]
        return Or(*expressions)

    def visit_Subtract(self, expression: "Subtract"):
        expr1 = self.visit(expression.expr1)
        expr2 = self.visit(expression.expr2)
        if isinstance(expr1, Constant) and isinstance(expr2, Constant):
            return expr1 - expr2
        return Subtract(expr1, expr2)

    def visit_Symbol(self, expression: "Symbol"):
        if expression.concrete_value is not None:
            return Constant(expression.concrete_value)
        return expression


class ToCNF(ExpressionTransformer):
    def visit_And(self, expression: "And"):
        expressions = []  # type: List[Expression]
        for expr in expression.expressions:
            expr = self.visit(expr)
            if isinstance(expr, And):
                expressions.extend(expr.expressions)
            else:
                expressions.append(expr)
        if len(expressions) == 1:
            return expressions[0]
        return And(*expressions)

    def visit_Forall(self, expression: "Forall"):
        expr = self.visit(expression.expression)
        return expr

    def visit_Implies(self, expression: "Implies"):
        return self.visit(Or(~expression.antecedent, expression.consequent))

    def visit_Or(self, expression: "Or"):
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
                return Constant(True)
            elif len(expressions) == 1:
                return expressions[0]
            return Or(*expressions)
        if len(expressions) == 0:
            return conjunction
        clauses = list(
            (e,) for e in conjunction.expressions
        )  # type: List[Tuple[Or, ...]]
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
