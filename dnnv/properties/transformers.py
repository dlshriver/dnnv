from typing import Dict, List, Optional, Tuple, Union

from .base import *
from .visitors import ExpressionVisitor


class ExpressionTransformer(ExpressionVisitor):
    def __init__(self):
        self.visited = {}

    def visit(self, expression: Expression):
        expression_id = hash(expression)
        if expression_id not in self.visited:
            method_name = "visit_%s" % expression.__class__.__name__
            visitor = getattr(self, method_name, self.generic_visit)
            self.visited[expression_id] = (visitor(expression), expression)
        return self.visited[expression_id][0]

    def generic_visit(self, expression: Expression):
        if isinstance(expression, Expression):
            raise ValueError(
                f"Unimplemented expression type: {type(expression).__name__}"
            )
        return self.visit(Constant(expression))


class GenericExpressionTransformer(ExpressionTransformer):
    def generic_visit(self, expression: Expression):
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


class PropagateConstants(ExpressionTransformer):
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
            if len(expressions) == 1:
                return expressions[0]
            return Add(*expressions)
        constant_value = 0
        for expr in constant_expressions:
            try:
                constant_value += expr.value
            except:
                print(constant_value, type(constant_value))
                print(expr.value, type(expr.value))
                print(type(expr))
                print(constant_value + expr.value)
                raise
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


class Simplify(GenericExpressionTransformer):
    def _extract_constants(self, expr: Add):
        if not isinstance(expr, Add):
            raise TypeError(
                "Argument 'expr' of '_extract_constants' must be an Add expression"
            )
        if not all(isinstance(e, Multiply) for e in expr.expressions):
            raise TypeError("Argument 'expr' must be sum of Multiply expression")
        constants = []
        expressions = expr.expressions  # type: List[Multiply]
        expr.expressions = []
        for e in expressions:
            if all(isinstance(m, Constant) for m in e.expressions):
                constants.append(e)
            else:
                expr.expressions.append(e)
        if len(constants) == 0:
            return expr, Constant(0)
        elif len(constants) == 1:
            return expr, constants[0]
        return expr, Add(*constants)

    def visit_Add(self, expression: Add) -> Add:
        summands = {}  # type: Dict[Expression, List[Expression]]
        expressions = [expr for expr in expression.expressions]
        while len(expressions):
            expr = self.visit(expressions.pop())
            if isinstance(expr, Add):
                expressions.extend(expr.expressions)
            elif isinstance(expr, Constant):
                one = Constant(1)
                if one not in summands:
                    summands[one] = []
                summands[one].append(expr)
            elif isinstance(expr, FunctionCall):
                if expr not in summands:
                    summands[expr] = []
                summands[expr].append(Constant(1))
            elif isinstance(expr, Multiply):
                cs = []
                vs = []
                for e in expr.expressions:
                    if isinstance(e, Constant):
                        cs.append(e)
                    elif not isinstance(e, (Subscript, Symbol)):
                        raise ValueError(
                            f"Unexpected type of operand for Multiply: {type(e).__name__!r}"
                        )
                    else:
                        vs.append(e)
                v = tuple(sorted(vs, key=lambda x: repr(x)))
                if len(v) == 0:
                    e = Constant(1)
                elif len(v) == 1:
                    e = v[0]
                else:
                    e = Multiply(*v)
                if len(cs) == 0:
                    c = Constant(1)  # type: Union[Constant, Multiply]
                elif len(cs) == 1:
                    c = cs[0]
                else:
                    c = Multiply(*cs)
                if e not in summands:
                    summands[e] = []
                summands[e].append(c)
            elif isinstance(expr, Negation):
                if expr.expr not in summands:
                    summands[expr.expr] = []
                summands[expr.expr].append(Constant(-1))
            elif isinstance(expr, Subscript):
                if expr not in summands:
                    summands[expr] = []
                summands[expr].append(Constant(1))
            elif isinstance(expr, Symbol):
                if expr not in summands:
                    summands[expr] = []
                summands[expr].append(Constant(1))
            else:
                raise ValueError(
                    f"Unexpected type of operand for Add: {type(expr).__name__!r}"
                )
        for var, coefs in summands.items():
            if len(coefs) == 1:
                expressions.append(coefs[0] * var)
            else:
                expressions.append(Add(*coefs) * var)
        return Add(*expressions)

    def visit_Equal(self, expression: Equal):
        lhs = self.visit(expression.expr1 - expression.expr2)
        lhs, nrhs = self._extract_constants(lhs)
        return Equal(lhs, -nrhs)

    def visit_GreaterThan(self, expression: GreaterThan) -> GreaterThan:
        lhs = self.visit(expression.expr1 - expression.expr2)
        lhs, nrhs = self._extract_constants(lhs)
        expr = GreaterThan(lhs, -nrhs)
        return expr

    def visit_GreaterThanOrEqual(
        self, expression: GreaterThanOrEqual
    ) -> GreaterThanOrEqual:
        lhs = self.visit(expression.expr1 - expression.expr2)
        lhs, nrhs = self._extract_constants(lhs)
        expr = GreaterThanOrEqual(lhs, -nrhs)
        return expr

    def visit_LessThan(self, expression: LessThan) -> GreaterThan:
        lhs = self.visit(expression.expr2 - expression.expr1)
        lhs, nrhs = self._extract_constants(lhs)
        expr = GreaterThan(lhs, -nrhs)
        return expr

    def visit_LessThanOrEqual(self, expression: LessThanOrEqual) -> GreaterThanOrEqual:
        lhs = self.visit(expression.expr2 - expression.expr1)
        lhs, nrhs = self._extract_constants(lhs)
        expr = GreaterThanOrEqual(lhs, -nrhs)
        return expr

    def visit_Negation(self, expression: Negation):
        expr = expression.expr
        if isinstance(expr, Add):
            expr.expressions = [-e for e in expr.expressions]
        elif isinstance(expr, Subtract):
            expr.expr1 = -expr.expr1
            expr.expr2 = -expr.expr2
        elif isinstance(expr, Multiply):
            expr.expressions.append(Constant(-1))
        elif isinstance(expr, Divide):
            if isinstance(expr.expr2, Constant):
                expr.expr2 = -expr.expr2
            else:
                expr.expr1 = -expr.expr1
        else:
            return -self.visit(expr)
        return self.visit(expr)

    def visit_NotEqual(self, expression: NotEqual):
        lhs = self.visit(expression.expr1 - expression.expr2)
        lhs, nrhs = self._extract_constants(lhs)
        return NotEqual(lhs, -nrhs)

    # def visit_Multiply(self, expression: Multiply):
    #     operands = [self.visit(expr) for expr in expression.expressions]
    #     additions = [expr for expr in operands if isinstance(expr, Add)]
    #     non_additions = [expr for expr in operands if not isinstance(expr, Add)]
    #     if len(additions) > 0:
    #         # TODO
    #         raise NotImplementedError()
    #     return Multiply(*operands)

    def visit_Subtract(self, expression: Subtract) -> Add:
        expr = self.visit(expression.expr1 + -expression.expr2)
        return expr


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
