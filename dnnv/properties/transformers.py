from types import new_class
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
        if not isinstance(expression, Expression):
            return self.generic_visit(expression)
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
            if isinstance(expression.path, Expression):
                path = self.visit(expression.path)
                return type(expression)(path)
            return expression
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

    def visit(self, expression: Expression) -> Expression:
        if self._top_level:
            expression = expression.propagate_constants()
            expression = SubstituteFunctionCalls(_BUILTIN_FUNCTIONS).visit(expression)
            expression = expression.propagate_constants()
            expression = expression.to_cnf()
            expression = expression.propagate_constants()
            # TODO : clean this up to not use internal implementation details
            to_cnf = ToCNF()
            to_cnf._top_level = False
            return to_cnf.visit(super().visit(expression))
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

    def visit_Equal(self, expression: Equal) -> And:
        expr1 = expression.expr1
        expr2 = expression.expr2
        return self.visit(And(expr1 <= expr2, expr2 <= expr1))

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

    def visit_Attribute(self, expression: Attribute) -> Attribute:
        expr1 = self.visit(expression.expr1)
        expr2 = self.visit(expression.expr2)
        expr = Attribute(expr1, expr2)
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
            if (
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
        return super().generic_visit(expression)

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
        if isinstance(expression, AssociativeExpression) and not isinstance(
            expression, (And, Or)
        ):
            assoc_expr = type(expression)
            exprs = [self.visit(expr) for expr in expression.expressions]
            if all(not isinstance(expr, IfThenElse) for expr in exprs):
                return assoc_expr(*exprs)
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
                    assoc_expr(ite_expr.t_expr, *expressions),
                    assoc_expr(ite_expr.f_expr, *expressions),
                )
            )
        elif isinstance(expression, BinaryExpression) and not isinstance(
            expression, Implies
        ):
            binary_expr = type(expression)
            expr1 = self.visit(expression.expr1)
            expr2 = self.visit(expression.expr2)
            if not isinstance(expr1, IfThenElse) and not isinstance(expr2, IfThenElse):
                return binary_expr(expr1, expr2)
            if isinstance(expr1, IfThenElse):
                return self.visit(
                    IfThenElse(
                        expr1.condition,
                        binary_expr(expr1.t_expr, expr2),
                        binary_expr(expr1.f_expr, expr2),
                    )
                )
            return self.visit(
                IfThenElse(
                    expr2.condition,
                    binary_expr(expr1, expr2.t_expr),
                    binary_expr(expr1, expr2.f_expr),
                )
            )
        elif isinstance(expression, UnaryExpression):
            unary_expr = type(expression)
            expr = self.visit(expression.expr)
            if not isinstance(expression.expr, IfThenElse):
                return unary_expr(expr)
            return self.visit(
                IfThenElse(
                    expr.condition, unary_expr(expr.t_expr), unary_expr(expr.f_expr)
                )
            )
        return super().generic_visit(expression)


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
            constant_value += expr.value
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
        if len(constant_expressions) == 0 and len(expressions) == 0:
            return Constant(True)
        constant_value = True
        for expr in constant_expressions:
            constant_value &= expr.value
        if not constant_value or len(expressions) == 0:
            return Constant(constant_value)
        if len(expressions) == 1:
            return expressions.pop()
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
            if isinstance(result, Expression):
                return result.propagate_constants()
            return Constant(result)
        new_function_call = FunctionCall(function, args, kwargs)
        if new_function_call.is_concrete:
            result = new_function_call.value
            if isinstance(result, Expression):
                return result.propagate_constants()
            return Constant(result)
        return new_function_call

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

    def visit_IfThenElse(self, expression: IfThenElse):
        condition = self.visit(expression.condition)
        t_expr = self.visit(expression.t_expr)
        f_expr = self.visit(expression.f_expr)
        if isinstance(condition, Constant):
            if condition.value:
                return t_expr
            return f_expr
        if (
            isinstance(t_expr, Constant)
            and isinstance(f_expr, Constant)
            and isinstance(t_expr.value, (bool, np.bool, np.bool_))
            and isinstance(f_expr.value, (bool, np.bool, np.bool_))
        ):
            if t_expr.value == True and f_expr.value == False:
                return condition
            elif t_expr.value == False and f_expr.value == True:
                return ~condition
            elif t_expr.value == True and f_expr.value == True:
                return Constant(True)
            elif t_expr.value == False and f_expr.value == False:
                return Constant(False)
        elif (
            isinstance(t_expr, Constant)
            and isinstance(t_expr.value, (bool, np.bool, np.bool_))
            and t_expr.value == False
        ):
            return And(~condition, f_expr)
        elif (
            isinstance(f_expr, Constant)
            and isinstance(f_expr.value, (bool, np.bool, np.bool_))
            and f_expr.value == False
        ):
            return And(condition, t_expr)
        if type(t_expr) == type(f_expr) and hash(t_expr) == hash(f_expr):
            return t_expr
        return IfThenElse(condition, t_expr, f_expr)

    def visit_Image(self, expression: Image):
        path = self.visit(expression.path)
        return Image.load(path)

    def visit_Implies(self, expression: Implies):
        antecedent = self.visit(expression.expr1)
        consequent = self.visit(expression.expr2)
        if isinstance(antecedent, Constant) and not antecedent.value:
            return Constant(True)
        elif isinstance(antecedent, Constant) and antecedent.value:
            return consequent
        elif isinstance(consequent, Constant) and consequent.value:
            return Constant(True)
        elif isinstance(consequent, Constant) and not consequent.value:
            return ~antecedent
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

    def visit_Not(self, expression: Network):
        expr = self.visit(expression.expr)
        if isinstance(expr, Constant):
            return Constant(~expr.value)
        return Not(expr)

    def visit_NotEqual(self, expression: NotEqual):
        expr1 = self.visit(expression.expr1)
        expr2 = self.visit(expression.expr2)
        if isinstance(expr1, Constant) and isinstance(expr2, Constant):
            return Constant(expr1.value != expr2.value)
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
        if len(constant_expressions) == 0 and len(expressions) == 0:
            return Constant(False)
        constant_value = False
        for expr in constant_expressions:
            constant_value |= expr.value
        if constant_value or len(expressions) == 0:
            return Constant(constant_value)
        if len(expressions) == 1:
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
    def visit(self, expression: Expression) -> Expression:
        if self._top_level:
            expression = LiftIfThenElse().visit(expression)
            expression = expression.propagate_constants()
            expression = RemoveIfThenElse().visit(expression)
            expression = expression.propagate_constants()
        expr = super().visit(expression)
        return expr

    def visit_And(self, expression: And) -> And:
        expressions = set()  # type: Set[Expression]
        for expr in expression.expressions:
            expr = self.visit(expr)
            if isinstance(expr, And):
                expressions = expressions.union(expr.expressions)
            else:
                expressions.add(expr)
        if len(expressions) == 1:
            return And(expressions.pop())
        return And(*expressions)

    def visit_Forall(self, expression: Forall):
        expr = self.visit(expression.expression)
        return expr

    def visit_Implies(self, expression: Implies) -> And:
        return self.visit(Or(~expression.expr1, expression.expr2))

    def visit_Or(self, expression: Or) -> And:
        expressions = set()  # type: Set[Expression]
        conjunction = None  # type: Optional[And]
        for expr in expression.expressions:
            expr = self.visit(expr)
            if isinstance(expr, Or):
                expressions = expressions.union(expr.expressions)
            elif isinstance(expr, And) and conjunction is None:
                conjunction = expr
            else:
                expressions.add(expr)
        if conjunction is None:
            if len(expressions) == 0:
                return And(Constant(True))
            elif len(expressions) == 1:
                return And(expressions.pop())
            return And(Or(*expressions))
        if len(expressions) == 0:
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
