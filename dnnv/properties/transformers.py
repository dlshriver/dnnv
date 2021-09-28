import numpy as np

from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Type, Union

from .base import *
from .base import _BUILTIN_FUNCTIONS, _BUILTIN_FUNCTION_TRANSFORMS
from .visitors import ExpressionVisitor


class ExpressionTransformer(ExpressionVisitor):
    def __init__(self):
        self.visited = {}
        self._top_level = True

    def visit(self, expression: Expression) -> Expression:
        if self._top_level:
            self._top_level = False
        if expression not in self.visited:
            method_name = "visit_%s" % expression.__class__.__name__
            visitor = getattr(self, method_name, self.generic_visit)
            self.visited[expression] = visitor(expression)
        return self.visited[expression]

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
            path = expression.path
            if isinstance(expression.path, Expression):
                path = self.visit(expression.path)
            return type(expression)(path)
        elif isinstance(expression, TernaryExpression):
            expr1 = self.visit(expression.expr1)
            expr2 = self.visit(expression.expr2)
            expr3 = self.visit(expression.expr3)
            return type(expression)(expr1, expr2, expr3)
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


class SubstituteFunctionCalls(GenericExpressionTransformer):
    def __init__(
        self, functions: Optional[Dict[Callable, Callable[..., Expression]]] = None
    ):
        super().__init__()
        if functions is None:
            self.functions = _BUILTIN_FUNCTIONS
        else:
            self.functions = functions
        self.function_transforms = _BUILTIN_FUNCTION_TRANSFORMS

    def generic_visit(self, expression: Expression):
        if isinstance(expression, BinaryExpression):
            return self.visit_BinaryExpression(expression)
        return super().generic_visit(expression)

    def visit_BinaryExpression(self, expression: BinaryExpression) -> BinaryExpression:
        expr_type = type(expression)
        expr1 = expression.expr1
        expr2 = expression.expr2
        if (
            isinstance(expr1, FunctionCall)
            and expr1.function.is_concrete
            and (expr1.function.value, expr_type) in self.function_transforms
        ):
            result = self.function_transforms[(expr1.function.value, expr_type)](
                expr1, expr2
            )
            if result is not NotImplemented:
                return self.visit(result)
        elif (
            isinstance(expr2, FunctionCall)
            and expr2.function.is_concrete
            and (expr2.function.value, expr_type) in self.function_transforms
        ):
            result = self.function_transforms[(expr2.function.value, expr_type)](
                expr2, expr1
            )
            if result is not NotImplemented:
                return self.visit(result)
        return expr_type(self.visit(expr1), self.visit(expr2))

    def visit_FunctionCall(self, expression: FunctionCall):
        function = self.visit(expression.function)
        args = tuple([self.visit(arg) for arg in expression.args])
        kwargs = {name: self.visit(value) for name, value in expression.kwargs.items()}
        if isinstance(function, Constant) and function.value in self.functions:
            return self.functions[function.value](*args, **kwargs)
        expr = FunctionCall(function, args, kwargs)
        return expr


class RemoveIfThenElse(GenericExpressionTransformer):
    def visit_IfThenElse(self, expression: IfThenElse):
        condition = self.visit(expression.condition)
        t_expr = self.visit(expression.t_expr)
        f_expr = self.visit(expression.f_expr)
        return And(Implies(condition, t_expr), Implies(~condition, f_expr))


class LiftIfThenElse(GenericExpressionTransformer):
    def generic_visit(self, expression: Expression) -> Expression:
        if isinstance(expression, AssociativeExpression):
            return self.visit_AssociativeExpression(expression)
        elif isinstance(expression, BinaryExpression):
            return self.visit_BinaryExpression(expression)
        elif isinstance(expression, UnaryExpression):
            return self.visit_UnaryExpression(expression)
        return super().generic_visit(expression)

    def visit_AssociativeExpression(
        self,
        expression: AssociativeExpression,
    ) -> AssociativeExpression:
        expr_t = type(expression)
        exprs = [self.visit(expr) for expr in expression.expressions]
        if all(not isinstance(expr, IfThenElse) for expr in exprs):
            return expr_t(*exprs)
        expressions = []
        ite_expr = None  # type: Optional[IfThenElse]
        for expr in exprs:
            if ite_expr is None and isinstance(expr, IfThenElse):
                ite_expr = expr
                continue
            expressions.append(expr)
        assert ite_expr is not None
        return self.visit(
            IfThenElse(
                ite_expr.condition,
                expr_t(ite_expr.t_expr, *expressions),
                expr_t(ite_expr.f_expr, *expressions),
            )
        )

    def visit_BinaryExpression(
        self,
        expression: BinaryExpression,
    ) -> BinaryExpression:
        expr_t = type(expression)
        expr1 = self.visit(expression.expr1)
        expr2 = self.visit(expression.expr2)
        if not isinstance(expr1, IfThenElse) and not isinstance(expr2, IfThenElse):
            return expr_t(expr1, expr2)
        if isinstance(expr1, IfThenElse):
            return self.visit(
                IfThenElse(
                    expr1.condition,
                    expr_t(expr1.t_expr, expr2),
                    expr_t(expr1.f_expr, expr2),
                )
            )
        return self.visit(
            IfThenElse(
                expr2.condition,
                expr_t(expr1, expr2.t_expr),
                expr_t(expr1, expr2.f_expr),
            )
        )

    def visit_UnaryExpression(
        self,
        expression: UnaryExpression,
    ) -> UnaryExpression:
        expr_t = type(expression)
        expr = self.visit(expression.expr)
        if not isinstance(expression.expr, IfThenElse):
            return expr_t(expr)
        return self.visit(
            IfThenElse(expr.condition, expr_t(expr.t_expr), expr_t(expr.f_expr))
        )


class PropagateConstants(ExpressionTransformer):
    def visit_Attribute(self, expression: Attribute) -> Union[Attribute, Constant]:
        expr = self.visit(expression.expr)
        name = self.visit(expression.name)
        if (
            isinstance(expr, Network)
            and expr.is_concrete
            and name.is_concrete
            and name.value == "compose"
        ):
            return Constant(expr.compose)
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
        constant_value = constant_expressions[0].value
        for expr in constant_expressions[1:]:
            constant_value = constant_value + expr.value
        if len(expressions) == 0:
            return Constant(constant_value)
        elif isinstance(constant_value, (float, int)) and constant_value == 0:
            return Add(*expressions)
        return Add(Constant(constant_value), *expressions)

    def visit_And(self, expression: And):
        expressions: Set[Expression] = set()
        constant_expressions = set()
        for expr in expression.expressions:
            expr = self.visit(expr)
            if isinstance(expr, Constant):
                constant_expressions.add(expr)
            elif (~expr) in expressions:
                return Constant(False)
            else:
                expressions.add(expr)
        if len(constant_expressions) == 0:
            constant_value = Constant(True)
        else:
            constant_value = constant_expressions.pop().value
            for expr in constant_expressions:
                constant_value = constant_value & expr.value
        if len(expressions) == 0:
            return Constant(constant_value)
        elif isinstance(constant_value, np.ndarray):
            if np.all(constant_value):
                if len(expressions) == 1:
                    return expressions.pop()
                else:
                    return And(*expressions)
            elif np.all(~constant_value):
                return Constant(False)
            return And(Constant(constant_value), *expressions)
        elif not constant_value:
            return Constant(constant_value)
        elif len(expressions) == 1:
            return expressions.pop()
        return And(*expressions)

    def visit_Constant(self, expression: Constant):
        return expression

    def visit_Divide(self, expression: Divide):
        expr1 = self.visit(expression.expr1)
        expr2 = self.visit(expression.expr2)
        if isinstance(expr1, Constant) and isinstance(expr2, Constant):
            return Constant(expr1.value / expr2.value)
        elif isinstance(expr2, Constant) and isinstance(expr2.value, (float, int)):
            if expr2.value == 1:
                return expr1
            elif expr2.value == -1:
                return -expr1
            elif expr2.value == 0:
                return Constant(float("nan"))
        elif (
            isinstance(expr1, Constant)
            and isinstance(expr1.value, (float, int))
            and expr1.value == 0
        ):
            return Constant(expr1.value)
        return Divide(expr1, expr2)

    def visit_Equal(self, expression: Equal):
        expr1 = self.visit(expression.expr1)
        expr2 = self.visit(expression.expr2)
        if isinstance(expr1, Constant) and isinstance(expr2, Constant):
            const_value = expr1.value == expr2.value
            if isinstance(const_value, np.ndarray):
                if np.all(const_value):
                    const_value = True
                elif np.all(~const_value):
                    const_value = False
            return Constant(const_value)
        elif expr1 is expr2:
            return Constant(True)
        return Equal(expr1, expr2)

    def visit_Exists(self, expression: Exists):
        variable = self.visit(expression.variable)
        expr = self.visit(expression.expression)
        if isinstance(expr, Constant):
            return Constant(bool(expr.value))
        return Exists(variable, expr)

    def visit_Forall(self, expression: Forall):
        variable = self.visit(expression.variable)
        expr = self.visit(expression.expression)
        if isinstance(expr, Constant):
            return Constant(bool(expr.value))
        elif variable not in expr.variables:
            return expr
        return Forall(variable, expr)

    def visit_FunctionCall(self, expression: FunctionCall):
        function = self.visit(expression.function)
        args = tuple([self.visit(arg) for arg in expression.args])
        kwargs = {name: self.visit(value) for name, value in expression.kwargs.items()}
        params = args + tuple(kwargs.values())
        if function.is_concrete and all(param.is_concrete for param in params):
            args = tuple(arg.value for arg in args)
            kwargs = {name: value.value for name, value in kwargs.items()}
            result = function.value(*args, **kwargs)
            if isinstance(result, Expression):
                return result.propagate_constants()
            return Constant(result)
        new_function_call = FunctionCall(function, args, kwargs)
        return new_function_call

    def visit_GreaterThan(self, expression: GreaterThan):
        expr1 = self.visit(expression.expr1)
        expr2 = self.visit(expression.expr2)
        if isinstance(expr1, Constant) and isinstance(expr2, Constant):
            const_value = expr1.value > expr2.value
            if isinstance(const_value, np.ndarray):
                if np.all(const_value):
                    const_value = True
                elif np.all(~const_value):
                    const_value = False
            return Constant(const_value)
        elif expr1 is expr2:
            return Constant(False)
        return GreaterThan(expr1, expr2)

    def visit_GreaterThanOrEqual(self, expression: GreaterThanOrEqual):
        expr1 = self.visit(expression.expr1)
        expr2 = self.visit(expression.expr2)
        if isinstance(expr1, Constant) and isinstance(expr2, Constant):
            const_value = expr1.value >= expr2.value
            if isinstance(const_value, np.ndarray):
                if np.all(const_value):
                    const_value = True
                elif np.all(~const_value):
                    const_value = False
            return Constant(const_value)
        elif expr1 is expr2:
            return Constant(True)
        return GreaterThanOrEqual(expr1, expr2)

    def visit_IfThenElse(self, expression: IfThenElse):
        condition = self.visit(expression.condition)
        t_expr = self.visit(expression.t_expr)
        f_expr = self.visit(expression.f_expr)
        bool_type = (bool, np.bool_)
        if condition.is_concrete:
            if condition.value:
                return t_expr
            return f_expr
        elif t_expr.is_concrete and f_expr.is_concrete:
            if (t_expr == f_expr).value:
                return Constant(t_expr)
            elif isinstance(t_expr.value, bool_type) and isinstance(
                f_expr.value, bool_type
            ):
                if t_expr.value:
                    return condition
                return ~condition
        return IfThenElse(condition, t_expr, f_expr)

    def visit_Image(self, expression: Image):
        path = self.visit(expression.path)
        return Image.load(path)

    def visit_Implies(self, expression: Implies):
        antecedent = self.visit(expression.expr1)
        consequent = self.visit(expression.expr2)
        if antecedent is consequent:
            return Constant(True)
        elif (
            antecedent.is_concrete
            and consequent.is_concrete
            and isinstance(antecedent.value, np.ndarray)
            and isinstance(consequent.value, np.ndarray)
        ):
            result = ~antecedent.value | consequent.value
            if np.all(result):
                return Constant(True)
            elif np.all(~result):
                return Constant(False)
            return Constant(result)
        elif antecedent.is_concrete:
            if isinstance(antecedent.value, np.ndarray):
                if np.all(~antecedent.value):
                    return Constant(True)
                elif np.all(antecedent.value):
                    return consequent
            elif not antecedent.value:
                return Constant(True)
            elif antecedent.value:
                return consequent
        elif consequent.is_concrete:
            if isinstance(consequent.value, np.ndarray):
                if np.all(~consequent.value):
                    return ~antecedent
                elif np.all(consequent.value):
                    return Constant(True)
            elif not consequent.value:
                return ~antecedent
            elif consequent.value:
                return Constant(True)
        return Implies(antecedent, consequent)

    def visit_LessThan(self, expression: LessThan):
        expr1 = self.visit(expression.expr1)
        expr2 = self.visit(expression.expr2)
        if isinstance(expr1, Constant) and isinstance(expr2, Constant):
            const_value = expr1.value < expr2.value
            if isinstance(const_value, np.ndarray):
                if np.all(const_value):
                    const_value = True
                elif np.all(~const_value):
                    const_value = False
            return Constant(const_value)
        elif expr1 is expr2:
            return Constant(False)
        return LessThan(expr1, expr2)

    def visit_LessThanOrEqual(self, expression: LessThanOrEqual):
        expr1 = self.visit(expression.expr1)
        expr2 = self.visit(expression.expr2)
        if isinstance(expr1, Constant) and isinstance(expr2, Constant):
            const_value = expr1.value <= expr2.value
            if isinstance(const_value, np.ndarray):
                if np.all(const_value):
                    const_value = True
                elif np.all(~const_value):
                    const_value = False
            return Constant(const_value)
        elif expr1 is expr2:
            return Constant(True)
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
        elif len(constant_expressions) == 0:
            if len(expressions) == 1:
                return expressions[0]
            return Multiply(*expressions)
        constant_value = constant_expressions[0].value
        for expr in constant_expressions[1:]:
            constant_value = constant_value * expr.value
        if np.all(constant_value == 0):
            return Constant(0)
        elif len(expressions) == 0:
            return Constant(constant_value)
        elif np.ndim(constant_value) == 0 and constant_value == 1:
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

    def visit_Not(self, expression: Network):
        expr = self.visit(expression.expr)
        bool_type = (bool, np.bool_)
        if isinstance(expr, Constant):
            if isinstance(expr.value, bool_type):
                return Constant(not expr.value)
            return Constant(~expr.value)
        return Not(expr)

    def visit_NotEqual(self, expression: NotEqual):
        expr1 = self.visit(expression.expr1)
        expr2 = self.visit(expression.expr2)
        if isinstance(expr1, Constant) and isinstance(expr2, Constant):
            const_value = expr1.value != expr2.value
            if isinstance(const_value, np.ndarray):
                if np.all(const_value):
                    const_value = True
                elif np.all(~const_value):
                    const_value = False
            return Constant(const_value)
        elif expr1 is expr2:
            return Constant(False)
        return NotEqual(expr1, expr2)

    def visit_Or(self, expression: Or):
        expressions: Set[Expression] = set()
        constant_expressions = set()
        for expr in expression.expressions:
            expr = self.visit(expr)
            if isinstance(expr, Constant):
                constant_expressions.add(expr)
            elif (~expr) in expressions:
                return Constant(True)
            else:
                expressions.add(expr)
        if len(constant_expressions) == 0:
            constant_value = Constant(False)
        else:
            constant_value = constant_expressions.pop().value
            for expr in constant_expressions:
                constant_value = constant_value | expr.value
        if len(expressions) == 0:
            return Constant(constant_value)
        elif isinstance(constant_value, np.ndarray):
            if np.all(~constant_value):
                if len(expressions) == 1:
                    return expressions.pop()
                else:
                    return Or(*expressions)
            elif np.all(constant_value):
                return Constant(True)
            return Or(Constant(constant_value), *expressions)
        elif constant_value:
            return Constant(constant_value)
        elif len(expressions) == 1:
            return expressions.pop()
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
        expr = self.visit(expression.expr1)
        index = self.visit(expression.expr2)
        if expr.is_concrete and index.is_concrete:
            if isinstance(expr, Network):
                return expr[index.value]
            return Constant(expr.value[index.value])
        return Subscript(expr, index)

    def visit_Subtract(self, expression: Subtract):
        expr1 = self.visit(expression.expr1)
        expr2 = self.visit(expression.expr2)
        if expr1 is expr2:
            return Constant(0)
        elif isinstance(expr1, Constant) and isinstance(expr2, Constant):
            return Constant(expr1.value - expr2.value)
        elif isinstance(expr1, Constant) and np.all(expr1.value == 0):
            return -expr2
        elif isinstance(expr2, Constant) and np.all(expr2.value == 0):
            return expr1
        return Subtract(expr1, expr2)

    def visit_Symbol(self, expression: Symbol):
        if expression.is_concrete:
            return Constant(expression.value)
        return expression


class ToCNF(GenericExpressionTransformer):
    def visit(self, expression: Expression) -> Expression:
        if self._top_level:
            expression = expression.propagate_constants()
            expression = LiftIfThenElse().visit(expression)
            expression = expression.propagate_constants()
            expression = RemoveIfThenElse().visit(expression)
            expression = expression.propagate_constants()
            expression = And(Or(expression))
        expr = super().visit(expression)
        return expr

    def visit_And(self, expression: And) -> And:
        expressions = set()  # type: Set[Expression]
        for expr in expression.expressions:
            expr = self.visit(expr)
            if isinstance(expr, And):
                expressions = expressions.union(expr.expressions)
            else:
                if not isinstance(expr, Or):
                    expr = Or(expr)
                expressions.add(expr)
        return And(*expressions)

    def visit_Exists(self, expression: Exists):
        raise NotImplementedError("Skolemization is not yet implemented.")

    def visit_Forall(self, expression: Forall):
        expr = self.visit(expression.expression)
        return expr

    def visit_Implies(self, expression: Implies) -> And:
        return self.visit(Or(~expression.expr1, expression.expr2))

    def visit_Or(self, expression: Or) -> And:
        conjunction: Optional[And] = None
        expressions: Set[Expression] = set()
        for expr in expression.expressions:
            expr = self.visit(expr)
            if conjunction is None and isinstance(expr, And):
                conjunction = expr
            else:
                expressions.add(expr)
        if conjunction is None:
            if len(expressions) == 0:
                return And(Or(Constant(False)))
            return And(Or(*expressions))
        elif len(expressions) == 0:
            return conjunction
        clauses = set(
            (e,) for e in conjunction.expressions
        )  # type: Set[Tuple[Expression, ...]]
        for expr in expressions:
            if isinstance(expr, And):
                new_clauses = set()
                for clause in clauses:
                    for e in expr.expressions:
                        new_clauses.add(tuple(set(clause + (e,))))
                clauses = new_clauses
            else:
                assert not isinstance(expr, Or)
                new_clauses = set()
                for clause in clauses:
                    new_clauses.add(tuple(set(clause + (expr,))))
                clauses = new_clauses
        return And(*[Or(*clause) for clause in clauses])


class Canonical(ToCNF):
    def _extract_constants(self, expr: Add) -> Tuple[Add, Constant]:
        constants = []
        expressions = expr.expressions
        expr.expressions = []
        for e in expressions:
            if e.is_concrete:
                constants.append(-e)
            else:
                expr.expressions.append(e)
        return expr, Add(*constants).propagate_constants()

    def visit(self, expression: Expression) -> Expression:
        if self._top_level:
            expression = expression.propagate_constants()
            expression = SubstituteFunctionCalls(_BUILTIN_FUNCTIONS).visit(expression)
        expr = super().visit(expression)
        return expr

    def visit_Add(self, expression: Add) -> Add:
        expressions = defaultdict(lambda: Constant(0))
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
            value = Constant(1)
            if symbol.is_concrete:
                symbol, value = value, symbol.propagate_constants()
            elif isinstance(expr, Multiply):
                constants = []
                symbols = []
                for e in expr.expressions:
                    if e.is_concrete:
                        constants.append(e)
                    else:
                        symbols.append(e)
                if len(symbols) == 1:
                    symbol = symbols[0]
                else:
                    symbol = Multiply(*symbols)
                value = Multiply(*constants).propagate_constants()
            expressions[symbol] = expressions[symbol] + value
        products = []
        for v, c in expressions.items():
            const = c.propagate_constants()
            if np.all(const.value == 0):
                continue
            products.append(Multiply(const, v))
        return Add(*products)

    def visit_Equal(self, expression: Equal) -> And:
        expr1 = expression.expr1
        expr2 = expression.expr2
        return self.visit(And(expr1 <= expr2, expr2 <= expr1))

    def visit_GreaterThan(self, expression: GreaterThan) -> GreaterThan:
        lhs = self.visit_Add(
            Add(expression.expr1, Multiply(Constant(-1), expression.expr2))
        )
        lhs, rhs = self._extract_constants(lhs)
        expr = GreaterThan(lhs, rhs)
        return expr

    def visit_GreaterThanOrEqual(
        self, expression: GreaterThanOrEqual
    ) -> GreaterThanOrEqual:
        lhs = self.visit_Add(
            Add(expression.expr1, Multiply(Constant(-1), expression.expr2))
        )
        lhs, rhs = self._extract_constants(lhs)
        expr = GreaterThanOrEqual(lhs, rhs)
        return expr

    def visit_LessThan(self, expression: LessThan) -> GreaterThan:
        lhs = self.visit_Add(
            Add(Multiply(Constant(-1), expression.expr1), expression.expr2)
        )
        lhs, rhs = self._extract_constants(lhs)
        expr = GreaterThan(lhs, rhs)
        return expr

    def visit_LessThanOrEqual(self, expression: LessThanOrEqual) -> GreaterThanOrEqual:
        lhs = self.visit_Add(
            Add(Multiply(Constant(-1), expression.expr1), expression.expr2)
        )
        lhs, rhs = self._extract_constants(lhs)
        expr = GreaterThanOrEqual(lhs, rhs)
        return expr

    def visit_Multiply(self, expression: Multiply) -> Add:
        expressions = [[]]  # type: List[List[Expression]]
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
            symbols = [Constant(1)]
            for e in expressions[0]:
                if e.is_concrete:
                    consts.append(e)
                else:
                    symbols.append(e)
            const = Multiply(*consts).propagate_constants()
            product = Multiply(*symbols).propagate_constants()
            return Add(Multiply(const, product))
        return self.visit(Add(*[Multiply(*e) for e in expressions]))

    def visit_Negation(self, expression: Negation) -> Add:
        return self.visit_Multiply(Multiply(Constant(-1), expression.expr))

    def visit_NotEqual(self, expression: NotEqual) -> Or:
        expr1 = expression.expr1
        expr2 = expression.expr2
        return self.visit(Or(expr1 > expr2, expr2 > expr1))

    def visit_Subtract(self, expression: Subtract) -> Add:
        return self.visit_Add(
            Add(expression.expr1, Multiply(Constant(-1), expression.expr2))
        )
