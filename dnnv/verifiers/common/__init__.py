from .errors import *
from .executors import *
from .results import *
from .utils import *

# TODO : move below to own package
import numpy as np

from collections import namedtuple
from enum import Enum
from typing import Iterable, Type, Union

from dnnv.properties import *


class Property:
    pass


class PropertyExtractor(ExpressionVisitor):
    def __init__(
        self, translator_error: Type[VerifierTranslatorError] = VerifierTranslatorError
    ):
        self.translator_error = translator_error
        self.initialize()

    def initialize(self):
        pass

    def extract(self) -> Property:
        return Property()

    def extract_from(self, expression: Expression) -> Iterable[Property]:
        self.initialize()
        self.visit(expression)
        yield self.extract()

    def visit(self, expression):
        method_name = "visit_%s" % expression.__class__.__name__
        visitor = getattr(self, method_name, None)
        if visitor is None:
            raise self.translator_error(
                "Unsupported property:"
                f" expression type {type(expression).__name__!r} is not currently supported"
            )
        return visitor(expression)


class IOProperty(Property):
    Constraint = namedtuple("Constraint", ["type", "idx", "coefs", "b"])
    ConstraintType = Enum("ConstraintType", ["LT", "LE", "EQ", "NE"])

    def __init__(
        self,
        network: Network,
        input_constraints: List[Constraint],
        output_constraints: List[Constraint],
    ):
        self.network = network
        self.input_constraints = input_constraints
        self.output_constraints = output_constraints


class LinIneqPropertyExtractor(PropertyExtractor):
    def initialize(self):
        self._stack = []
        self._constraint_type = None
        self.input = None
        self.network = None
        self.input_constraints = []
        self.output_constraints = []

    def extract(self) -> IOProperty:
        return IOProperty(self.network, self.input_constraints, self.output_constraints)

    def extract_from(self, expression: Expression) -> Iterable[IOProperty]:
        self.existential = False
        if isinstance(expression, Exists):
            expression = ~expression
            self.existential = True
        expression = expression.to_cnf()
        expression = expression.propagate_constants()
        expression = expression.simplify()
        expression = expression.propagate_constants()
        print("\nCNF:", expression, end="\n\n")
        not_expression = Or(~expression)
        for conjunction in not_expression:
            print("CHECKSAT:", conjunction)
            if len(conjunction.networks) != 1:
                raise self.translator_error(
                    "Exactly one network input and output are required"
                )
            yield from super().extract_from(conjunction)
            print()

    def visit(self, expression: Expression):
        self._stack.append(type(expression))
        if isinstance(expression, Constant):
            result = expression
        else:
            result = super().visit(expression)
        self._stack.pop()
        return result

    def visit_Add(self, expression: Add):
        if len(self._stack) != 3:
            raise self.translator_error("Property formula was not properly simplified")
        indices = []
        coefs = []
        for expr in expression.expressions:
            i, coef = self.visit(expr)
            if isinstance(i, Expression) or isinstance(coef, Expression):
                raise self.translator_error(
                    "Unsupported property: Symbolic values in Add expression"
                )
            indices.append(i)
            coefs.append(coef)
        return indices, coefs

    def visit_And(self, expression: And):
        if len(self._stack) != 1:
            raise self.translator_error("Property formula was not properly simplified")
        for expr in expression.expressions:
            constraints = self.visit(expr)
            if len(expr.networks) == 0:
                self.input_constraints.extend(constraints)
            else:
                self.output_constraints.extend(constraints)
        return expression

    def visit_FunctionCall(self, expression: FunctionCall):
        function = self.visit(expression.function)
        args = [self.visit(arg) for arg in expression.args]
        kwargs = {k: self.visit(v) for k, v in expression.kwargs.items()}
        return expression

    def _build_constraint(
        self, cmp_expr: Union[LessThan, LessThanOrEqual, Equal, NotEqual]
    ):
        if len(self._stack) != 2:
            raise self.translator_error("Property formula was not properly simplified")
        rhs = self.visit(cmp_expr.expr2)
        if not isinstance(rhs, Constant):
            raise self.translator_error("Property formula was not properly simplified")
        lhs = self.visit(cmp_expr.expr1)
        constraints = []
        if self._constraint_type is None:
            raise self.translator_error("Invalid constraint type: None specified")
        if isinstance(cmp_expr.expr1, Add):
            constraints.append(
                IOProperty.Constraint(self._constraint_type, lhs[0], lhs[1], rhs.value)
            )
        elif isinstance(cmp_expr.expr1, Multiply):
            idx, coefs = lhs
            if isinstance(idx, Symbol):
                for i in np.ndindex(rhs.value.shape):
                    constraints.append(
                        IOProperty.Constraint(
                            self._constraint_type, [i], [coefs], rhs.value[i]
                        )
                    )
            elif isinstance(idx, Expression):
                raise self.translator_error(
                    "Unsupported property: Symbolic values in Multiply expression"
                )
            else:
                constraints.append(
                    IOProperty.Constraint(
                        self._constraint_type, [idx], [coefs], rhs.value
                    )
                )
        elif isinstance(cmp_expr.expr1, Symbol):
            idx, coefs = lhs
            for i in np.ndindex(rhs.value.shape):
                constraints.append(
                    IOProperty.Constraint(
                        self._constraint_type, [i], [coefs], rhs.value[i]
                    )
                )
        else:
            raise self.translator_error(
                "Unsupported property:"
                f" Unexpected expression of type {type(cmp_expr.expr1).__name__!r}"
                f" in left hand side of {type(cmp_expr).__name__}"
            )
        return constraints

    def visit_LessThan(self, expression: LessThan):
        self._constraint_type = IOProperty.ConstraintType.LT
        return self._build_constraint(expression)

    def visit_LessThanOrEqual(self, expression: LessThanOrEqual):
        self._constraint_type = IOProperty.ConstraintType.LE
        return self._build_constraint(expression)

    def visit_Multiply(self, expression: Multiply):
        if len(expression.expressions) != 2:
            raise self.translator_error("Property formula was not properly simplified")
        if isinstance(expression.expressions[0], Constant):
            const = self.visit(expression.expressions[0]).value
            index, coef = self.visit(expression.expressions[1])
        elif isinstance(expression.expressions[1], Constant):
            const = self.visit(expression.expressions[1]).value
            index, coef = self.visit(expression.expressions[0])
        else:
            raise self.translator_error(
                "Unsupported property: Multiplication by non-constant value"
            )
        return index, coef * const

    def visit_Network(self, expression: Network):
        if self.network is None:
            self.network = expression
        elif self.network is not expression:
            raise self.translator_error("Multiple networks detected in property")
        return expression

    def visit_NotEqual(self, expression: NotEqual):
        self._constraint_type = IOProperty.ConstraintType.NE
        return self._build_constraint(expression)

    def visit_Subscript(self, expression: Subscript):
        index = self.visit(expression.index)
        if not isinstance(index, Constant):
            raise self.translator_error("Property formula was not properly simplified")
        expr = self.visit(expression.expr)
        if isinstance(expr, FunctionCall):
            if isinstance(expr.function, Network):
                value = 1
            else:
                raise self.translator_error(
                    "Unsupported property:"
                    f" Calling {type(expr).__name__!r} expressions is not currently supported"
                )
        else:
            raise self.translator_error(
                "Unsupported property:"
                f" Subscript is not currently supported for {type(expr).__name__!r} expressions"
            )
        return index.value, value

    def visit_Symbol(self, expression: Symbol):
        if self.input is None:
            self.input = expression
        elif self.input is not expression:
            raise self.translator_error("Multiple inputs detected in property")
        return expression, 1

