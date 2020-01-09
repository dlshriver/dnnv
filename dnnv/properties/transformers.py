from collections import defaultdict
from typing import Dict, List, Optional, Tuple, Union

from .base import *
from .visitors import ExpressionVisitor


class ExpressionTransformer(ExpressionVisitor):
    def __init__(self):
        self.visited = {}

    def visit(self, expression: Expression) -> Expression:
        expression_id = hash(expression)
        if expression_id not in self.visited:
            method_name = "visit_%s" % expression.__class__.__name__
            visitor = getattr(self, method_name, self.generic_visit)
            self.visited[expression_id] = (visitor(expression), expression)
        return self.visited[expression_id][0]

    def generic_visit(self, expression: Expression) -> Expression:
        if isinstance(expression, Expression):
            raise ValueError(
                f"Unimplemented expression type: {type(expression).__name__}"
            )
        return self.visit(Constant(expression))


class GenericExpressionTransformer(ExpressionTransformer):
    def generic_visit(self, expression: Expression) -> Expression:
        if isinstance(expression, AssociativeExpression):
            exprs = [self.visit(expr) for expr in expression.expressions]
            return type(expression)(*exprs)
        elif isinstance(expression, BinaryExpression):
            expr1 = self.visit(expression.expr1)
            expr2 = self.visit(expression.expr2)
            return type(expression)(expr1, expr2)
        elif isinstance(expression, Constant):
            return Constant(expression.value)
        elif isinstance(expression, FunctionCall):
            function = self.visit(expression.function)
            args = tuple(self.visit(arg) for arg in expression.args)
            kwargs = {k: self.visit(v) for k, v in expression.kwargs.items()}
            return FunctionCall(function, args, kwargs)
        elif isinstance(expression, Image):
            if isinstance(expression.path, Expression):
                path = self.visit(expression.path)
                return type(expression)(path)
            return expression
        elif isinstance(expression, Quantifier):
            variable = self.visit(expression.variable)
            expr = self.visit(expression.expression)
            return type(expression)(variable, expr)
        elif isinstance(expression, Symbol):
            return expression
        elif isinstance(expression, UnaryExpression):
            expr = self.visit(expression.expr)
            return type(expression)(expr)
        return super().generic_visit(expression)


class Canonical(ExpressionTransformer):
    def _extract_cmp_constants(self, expr: Add):
        constants = []
        expressions = expr.expressions
        expr.expressions = []
        for e in expressions:
            if e.is_concrete:
                constants.append(-e)
            else:
                expr.expressions.append(e)
        return expr, Add(*constants).propagate_constants()

    def __init__(self):
        super().__init__()
        self._top = True

    def visit(self, expression: Expression) -> Expression:
        top = False
        if self._top:
            self._top = False
            expression = expression.to_cnf()
            expression = expression.propagate_constants()
        expr = super().visit(expression)
        return expr

    def visit_And(self, expression: And) -> And:
        expressions = []
        for expr in expression.expressions:
            expr = self.visit(expr)
            if isinstance(expr, And):
                expressions.extend(expr.expressions)
            else:
                expressions.append(Or(expr))
        return And(*expressions)

    def visit_Or(self, expression: Or) -> Or:
        expressions = []
        for expr in expression.expressions:
            expr = self.visit(expr)
            if isinstance(expr, Or):
                expressions.extend(expr.expressions)
            else:
                expressions.append(expr)
        return Or(*expressions)

    def visit_GreaterThan(self, expression: GreaterThan) -> GreaterThan:
        lhs = self.visit(expression.expr1 - expression.expr2)
        if not isinstance(lhs, Add):
            raise RuntimeError(
                f"Expected left hand side of {type(expression).__name__!r} to be of type 'Add'"
            )
        lhs, rhs = self._extract_cmp_constants(lhs)
        expr = GreaterThan(lhs, rhs)
        return expr

    def visit_GreaterThanOrEqual(
        self, expression: GreaterThanOrEqual
    ) -> GreaterThanOrEqual:
        lhs = self.visit(expression.expr1 - expression.expr2)
        if not isinstance(lhs, Add):
            raise RuntimeError(
                f"Expected left hand side of {type(expression).__name__!r} to be of type 'Add'"
            )
        lhs, rhs = self._extract_cmp_constants(lhs)
        expr = GreaterThanOrEqual(lhs, rhs)
        return expr

    def visit_LessThan(self, expression: LessThan) -> GreaterThan:
        lhs = self.visit(-expression.expr1 + expression.expr2)
        if not isinstance(lhs, Add):
            raise RuntimeError(
                f"Expected left hand side of {type(expression).__name__!r} to be of type 'Add'"
            )
        lhs, rhs = self._extract_cmp_constants(lhs)
        expr = GreaterThan(lhs, rhs)
        return expr

    def visit_LessThanOrEqual(self, expression: LessThanOrEqual) -> GreaterThanOrEqual:
        lhs = self.visit(-expression.expr1 + expression.expr2)
        if not isinstance(lhs, Add):
            raise RuntimeError(
                f"Expected left hand side of {type(expression).__name__!r} to be of type 'Add'"
            )
        lhs, rhs = self._extract_cmp_constants(lhs)
        expr = GreaterThanOrEqual(lhs, rhs)
        return expr

    def visit_Add(self, expression: Add) -> Add:
        expressions = defaultdict(list)  # type: Dict[Expression, List[Expression]]
        operands = [e for e in expression.expressions]
        while len(operands):
            expr = self.visit(operands.pop())
            if expr.is_concrete:
                expressions[Constant(1)].append(expr)
            elif isinstance(expr, Add):
                operands.extend(expr.expressions)
            elif isinstance(expr, Multiply):
                constants = []
                symbols = []
                for e in expr.expressions:
                    if e.is_concrete:
                        constants.append(e)
                    else:
                        symbols.append(e)
                symbol = Multiply(*tuple(sorted(symbols, key=repr)))
                expressions[symbol].append(Multiply(*constants))
            else:
                expressions[expr].append(Constant(1))
        products = []
        for v, c in expressions.items():
            const = Add(*c).propagate_constants()
            if isinstance(const.value, (int, float)) and const.value == 0:
                continue
            products.append(Multiply(const, v))
        return Add(*products)

    def visit_Multiply(self, expression: Multiply) -> Union[Add, Multiply]:
        expressions = [[]]  # type: List[List[Expression]]
        for expr in expression.expressions:
            expr = self.visit(expr)
            if isinstance(expr, Multiply):
                raise NotImplementedError()
            elif isinstance(expr, Add):
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
        if len(expressions) > 1:
            return Add(*[Multiply(*e) for e in expressions])
        return Multiply(*expressions[0])

    def visit_Negation(self, expression: Negation) -> Union[Add, Constant, Multiply]:
        if isinstance(expression.expr, Constant):
            return Constant(-self.visit(expression.expr).value)
        expr = self.visit(Multiply(Constant(-1), expression.expr))
        if not isinstance(expr, (Add, Multiply)):
            raise RuntimeError("Expected expression of type 'Add' or 'Multiply'")
        return expr

    def visit_Subtract(self, expression: Subtract) -> Add:
        expr = self.visit(expression.expr1 + -expression.expr2)
        if not isinstance(expr, Add):
            raise RuntimeError("Expected expression of type 'Add'")
        return expr

    def visit_Constant(self, expression: Constant) -> Constant:
        return expression

    def visit_FunctionCall(self, expression: FunctionCall) -> FunctionCall:
        function = self.visit(expression.function)
        args = tuple([self.visit(arg) for arg in expression.args])
        kwargs = {name: self.visit(value) for name, value in expression.kwargs.items()}
        expr = FunctionCall(function, args, kwargs)
        return expr

    def visit_Network(self, expression: Network) -> Network:
        return expression

    def visit_Subscript(self, expression: Subscript) -> Union[Constant, Subscript]:
        expr = self.visit(expression.expr1)[self.visit(expression.expr2)]
        return expr

    def visit_Symbol(self, expression: Symbol) -> Symbol:
        return expression


class PropagateConstants(ExpressionTransformer):
    def visit_Attribute(self, expression: Attribute) -> Union[Attribute, Constant]:
        expr = self.visit(expression.expr)
        name = self.visit(expression.name)
        if expr.is_concrete and name.is_concrete:
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
            if len(expressions) == 1:
                return expressions[0]
            return Add(*expressions)
        constant_value = 0
        for expr in constant_expressions:
            constant_value += expr.value
        if len(expressions) == 0:
            return Constant(constant_value)
        elif isinstance(constant_value, (float, int)) and constant_value == 0:
            return Add(*expressions)
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
        elif (
            isinstance(expr1, Constant)
            and isinstance(expr1.value, (float, int))
            and expr1.value == 0
        ):
            return Constant(0)
        elif isinstance(expr2, Constant) and isinstance(expr2.value, (float, int)):
            if expr2.value == 1:
                return expr1
            elif expr2.value == 0:
                return Constant(float("nan"))
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
            return Constant(bool(expr.value))
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
        if isinstance(constant_value, (float, int)) and constant_value == 0:
            return Constant(0)
        elif len(expressions) == 0:
            return Constant(constant_value)
        elif isinstance(constant_value, (float, int)) and constant_value == 1:
            if len(expressions) > 1:
                return Multiply(*expressions)
            else:
                return expressions[0]
        return Multiply(Constant(constant_value), *expressions)

    def visit_Negation(self, expression: Negation):
        expr = self.visit(expression.expr)
        if isinstance(expr, Constant):
            return Constant(-expr.value)
        return -expr

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

    def visit_Slice(self, expression: Slice):
        start = self.visit(expression.start)
        stop = self.visit(expression.stop)
        step = self.visit(expression.step)
        if (
            isinstance(start, Constant)
            and isinstance(stop, Constant)
            and isinstance(step, Constant)
        ):
            return Constant(slice(start.value, stop.value, step.value))
        return Slice(start, stop, step)

    def visit_Subscript(self, expression: Subscript):
        expr = self.visit(expression.expr)
        index = self.visit(expression.index)
        if isinstance(expr, Constant) and isinstance(index, Constant):
            return Constant(expr.value[index.value])
        elif (
            isinstance(expr, Network)
            and expr.is_concrete
            and isinstance(index, Constant)
        ):
            return expr[index]
        return Subscript(expr, index)

    def visit_Subtract(self, expression: Subtract):
        expr1 = self.visit(expression.expr1)
        expr2 = self.visit(expression.expr2)
        if isinstance(expr1, Constant) and isinstance(expr2, Constant):
            return Constant(expr1.value - expr2.value)
        elif (
            isinstance(expr1, Constant)
            and isinstance(expr1.value, (float, int))
            and expr1.value == 0
        ):
            return -expr2
        elif (
            isinstance(expr2, Constant)
            and isinstance(expr2.value, (float, int))
            and expr2.value == 0
        ):
            return expr1
        return Subtract(expr1, expr2)

    def visit_Symbol(self, expression: Symbol):
        if expression.is_concrete:
            return Constant(expression.value)
        return expression


class ToCNF(GenericExpressionTransformer):
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
