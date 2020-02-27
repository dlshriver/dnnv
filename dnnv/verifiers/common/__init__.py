from .errors import *
from .executors import *
from .results import *
from .utils import *

# TODO : move below to own package
import numpy as np

from abc import ABC
from collections import namedtuple
from enum import Enum
from typing import Dict, Iterable, Tuple, Type, Union

from dnnv import logging
from dnnv.properties import *

MAGIC_NUMBER = 1e-12


class Constraint(ABC):
    @property
    def is_consistent(self):
        return None

    @abstractmethod
    def as_layers(
        self,
        network: Network,
        layer_types: Optional[List[Type[Layer]]] = None,
        extra_layer_types: Optional[List[Type[Layer]]] = None,
        translator_error: Type[VerifierTranslatorError] = VerifierTranslatorError,
    ) -> List[Layer]:
        raise NotImplementedError()


class ConvexPolytope(Constraint):
    LinearInequality = namedtuple("LinearInequality", ["indices", "coefficients", "b"])

    def __init__(self, shape):
        self.shape = shape
        self.constraints = []  # type: List[LinearInequality]

    @property
    def is_consistent(self):
        from scipy.optimize import linprog
        from itertools import chain

        k = len(self.constraints)
        v = {}
        for c in self.constraints:
            for i in c.indices:
                if i not in v:
                    v[i] = len(v)
        n = len(v)
        A = np.zeros((k, n))
        b = np.zeros(k)
        for ci, c in enumerate(self.constraints):
            for i, a in zip(c.indices, c.coefficients):
                A[ci, v[i]] = a
            b[ci] = c.b
        obj = np.zeros(n)
        result = linprog(obj, A, b, bounds=(None, None))
        if result.status == 2:  # infeasible
            return False
        elif result.status == 0:  # feasible
            return True
        return None  # unknown

    def add_constraint(self, indices, coefficients, b):
        self.constraints.append(self.LinearInequality(indices, coefficients, b))

    def as_hyperrectangle(self):
        hr = HyperRectangle(self.shape)
        for c in self.constraints:
            hr.add_constraint(*c)
        return hr

    def as_layers(
        self,
        network: Network,
        layer_types: Optional[List[Type[Layer]]] = None,
        extra_layer_types: Optional[List[Type[Layer]]] = None,
        translator_error: Type[VerifierTranslatorError] = VerifierTranslatorError,
    ) -> List[Layer]:
        layers = as_layers(
            network.value,
            layer_types=layer_types,
            extra_layer_types=extra_layer_types,
            translator_error=translator_error,
        )
        last_layer = layers[-1]
        if not isinstance(last_layer, FullyConnected):
            # TODO
            raise translator_error(
                "Expected last layer of network to be fully connected"
            )
        last_layer_size = last_layer.bias.shape[0]
        new_layer_size = len(self.constraints)
        W = np.zeros((last_layer_size, new_layer_size))
        b = np.zeros(new_layer_size)
        for n, c in enumerate(self.constraints):
            b[n] = -c.b
            for i, v in zip(c.indices, c.coefficients):
                assert len(i) == 2 and i[0] == 0
                W[i[1], n] = v
        if last_layer.activation is None:
            last_layer.weights = last_layer.weights @ W
            last_layer.bias = last_layer.bias @ W + b
            last_layer.activation = "relu"
        else:
            last_layer = FullyConnected(W, b, activation="relu")
            layers.append(last_layer)
        if new_layer_size > 1:
            W_out = np.zeros((new_layer_size, 1))
            for i in range(new_layer_size):
                W_out[i, 0] = 1
            b_out = np.zeros(1)
            layers.append(FullyConnected(W_out, b_out))
        else:
            last_layer.activation = None
        return layers


class HyperRectangle(ConvexPolytope):
    def __init__(self, shape):
        super().__init__(shape)
        self._constraint_index = {}
        self.lower_bound = np.zeros(shape) - np.inf
        self.upper_bound = np.zeros(shape) + np.inf

    @property
    def is_consistent(self):
        if (self.lower_bound > self.upper_bound).any():
            return False
        return True

    def add_constraint(self, indices, coefficients, b):
        if len(indices) > 1:
            raise ValueError(
                "HyperRectangle constraints can only constrain a single dimension"
            )

        coef = np.sign(coefficients[0])
        if coef < 0:
            self.lower_bound[indices[0]] = max(
                b / coefficients[0], self.lower_bound[indices[0]]
            )
        elif coef > 0:
            self.upper_bound[indices[0]] = min(
                b / coefficients[0], self.upper_bound[indices[0]]
            )

        c_index = (indices[0], coef)
        if c_index in self._constraint_index:
            i = self._constraint_index[c_index]
            if self.constraints[i].b > (b / abs(coefficients[0])):
                self.constraints[i] = self.LinearInequality(
                    indices, [coef], b / abs(coefficients[0])
                )
        else:
            self._constraint_index[c_index] = len(self.constraints)
            self.constraints.append(
                self.LinearInequality(indices, [coef], b / abs(coefficients[0]))
            )

        return self


class Property:
    def __init__(
        self,
        network: Network,
        input_constraint: Constraint,
        output_constraint: Constraint,
    ):
        self.network = network
        self.input_constraint = input_constraint
        self.output_constraint = output_constraint


class PropertyExtractor(ExpressionVisitor):
    def __init__(
        self, translator_error: Type[VerifierTranslatorError] = VerifierTranslatorError
    ):
        self.translator_error = translator_error
        self.initialize()

    def initialize(self):
        pass

    def extract(self) -> Property:
        raise NotImplementedError()

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


class ConvexPolytopeExtractor(PropertyExtractor):
    def __init__(self):
        super().__init__()
        self._network_input_shapes = {}

    def initialize(self):
        self._stack = []
        self._network_output_shape = None
        self.input = None
        self.network = None
        self.input_constraint = None
        self.output_constraint = None

    def extract(self) -> Property:
        return Property(self.network, self.input_constraint, self.output_constraint)

    def extract_from(self, expression: Expression) -> Iterable[Property]:
        logger = logging.getLogger(__name__)
        self.existential = False
        if isinstance(expression, Exists):
            raise NotImplementedError()  # TODO
            expression = ~expression
            self.existential = True
        expression = expression.canonical()
        not_expression = Or(~expression)
        for conjunction in not_expression:
            logger.info("CONJUNCTION: %s", conjunction)
            if len(conjunction.networks) != 1:
                continue
                raise self.translator_error(
                    "Exactly one network input and output are required"
                )
            if len(conjunction.variables) != 1:
                raise self.translator_error("Exactly one network input is required")
            for prop in super().extract_from(conjunction):
                if prop.input_constraint.as_hyperrectangle().is_consistent == False:
                    continue
                if prop.output_constraint.is_consistent == False:
                    continue
                yield prop

    def visit(self, expression: Expression):
        self._stack.append(type(expression))
        if isinstance(expression, Constant):
            result = expression
        else:
            result = super().visit(expression)
        self._stack.pop()
        return result

    def visit_Add(
        self, expression: Add
    ) -> Tuple[List[Union[Dict[str, Tuple[int, ...]], np.ndarray]], List[float]]:
        if len(self._stack) > 3:
            raise self.translator_error(
                "Not Canonical:"
                f" {type(expression).__name__!r} expression found below expected level"
            )
        indices = []  # type: List[Union[Dict[str, Tuple[int, ...]], np.ndarray]]
        coefs = []
        for expr in expression.expressions:
            i, coef = self.visit(expr)
            if isinstance(i, Expression) or isinstance(coef, Expression):
                raise self.translator_error(
                    "Unsupported property: Symbolic values in Add expression"
                )
            if len(indices) > 0:
                if type(i) != type(indices[0]):
                    raise self.translator_error(
                        "Invalid property: Adding expressions with different types is not supported"
                    )
                elif (
                    isinstance(i, np.ndarray)
                    and isinstance(indices[0], np.ndarray)
                    and i.shape != indices[0].shape
                ):
                    raise self.translator_error(
                        "Invalid property: Adding expressions with different shapes is not supported"
                    )
            indices.append(i)
            coefs.append(coef)
        return indices, coefs

    def visit_And(self, expression: And):
        if len(self._stack) != 1:
            raise self.translator_error(
                "Not Canonical: 'And' expression not at top level"
            )
        for expr in sorted(expression.expressions, key=lambda e: -len(e.networks)):
            self.visit(expr)
        return expression

    def visit_FunctionCall(self, expression: FunctionCall):
        if isinstance(expression.function, Network):
            function = self.visit(expression.function)
            input_details = expression.function.value.input_details
            if len(expression.args) != len(input_details):
                raise self.translator_error(
                    "Invalid property:"
                    f" Not enough inputs for network '{expression.function}'"
                )
            for arg, d in zip(expression.args, input_details):
                if arg in self._network_input_shapes:
                    if self._network_input_shapes[arg] != tuple(d.shape):
                        raise self.translator_error(
                            f"Invalid property: variable with multiple shapes: '{arg}'"
                        )
                self._network_input_shapes[arg] = tuple(d.shape)
            args = [self.visit(arg) for arg in expression.args]
            if len(expression.kwargs) > 0:
                raise self.translator_error(
                    "Unsupported property:"
                    f" Executing networks with keyword arguments is not currently supported"
                )

            shape = self._network_output_shape
            return (
                np.array([{"index": i} for i in np.ndindex(shape)]).reshape(shape),
                np.ones(shape),
            )
        raise self.translator_error(
            "Unsupported property:"
            f" Function {expression.function} is not currently supported"
        )

    def _add_constraint(self, expr: Union[LessThan, LessThanOrEqual]):
        if len(self._stack) > 2:
            raise self.translator_error(
                f"Not Canonical: {type(expr).__name__!r} expression below expected level"
            )
        if not isinstance(expr.expr1, Add):
            raise self.translator_error(
                "Not Canonical:"
                f" LHS of {type(expr).__name__!r} is not an 'Add' expression"
            )
        if not isinstance(expr.expr2, Constant):
            raise self.translator_error(
                "Not Canonical:"
                f" RHS of {type(expr).__name__!r} is not a 'Constant' expression"
            )
        lhs = self.visit(expr.expr1)
        rhs = self.visit(expr.expr2)

        indices, coefs = lhs
        constraints = []  # type: List[Tuple[List[Tuple[int, ...]], List[float], float]]
        for idx, coef in zip(indices, coefs):
            if isinstance(idx, np.ndarray):
                if len(constraints) > 0:
                    # TODO : needs a better error message
                    raise self.translator_error("Unsupported property: ?")
                for i, c in zip(idx.flatten(), coef.flatten()):
                    index = i["index"]
                    if isinstance(rhs.value, np.ndarray):
                        constraints.append(([index], [c], rhs.value[index]))
                    elif isinstance(rhs.value, (int, float)):
                        constraints.append(([index], [c], rhs.value))
                    else:
                        raise self.translator_error(
                            "Unsupported property:"
                            " Unexpected type for right hand side of comparison:"
                            f" {type(rhs.value).__name__!r}"
                        )
            elif isinstance(idx, dict):
                if len(constraints) == 0:
                    constraints.append(([], [], rhs.value))
                elif len(constraints) > 1:
                    # TODO : needs a better error message
                    raise self.translator_error("Unsupported property: ?")
                c = constraints[0]
                index = idx["index"]
                if index in c[0]:
                    c[1][c[0].index(index)] += coef
                else:
                    c[0].append(idx["index"])
                    c[1].append(coef)

        if len(expr.networks) == 0:
            current_constraint = self.input_constraint
        else:
            current_constraint = self.output_constraint
        for (i, c, b) in constraints:
            if isinstance(expr, LessThanOrEqual):
                current_constraint.add_constraint(i, c, b)
            elif isinstance(expr, LessThan):
                # current_constraint.add_constraint(i, c, np.nextafter(b, -1))
                current_constraint.add_constraint(
                    i, c, b - 1e-6
                )  # TODO : can we remove this magic number? maybe make it parameterizable?
            else:
                raise self.translator_error(
                    "Unsupported property:"
                    f" {type(expr).__name__!r} expressions are not yet supported"
                )
        return expr

    def visit_LessThan(self, expression: LessThan):
        return self._add_constraint(expression)

    def visit_LessThanOrEqual(self, expression: LessThanOrEqual):
        return self._add_constraint(expression)

    def visit_Multiply(self, expression: Multiply):
        constants = []
        symbols = []
        for expr in expression.expressions:
            expr = self.visit(expr)
            if isinstance(expr, Expression) and expr.is_concrete:
                constants.append(expr.value)
            else:
                symbols.append(expr)
        if len(symbols) > 1:
            raise self.translator_error(
                "Unsupported property: Multiplication of symbolic values"
            )
        index, coef = symbols[0]
        return index, np.product(constants) * coef

    def visit_Network(self, expression: Network):
        if self.network is None:
            self.network = expression
            if len(expression.value.output_operations) > 1:
                raise NotImplementedError(
                    "Networks with multiple output operations are not currently supported"
                )
            self._network_output_shape = expression.value.output_shape[0]
            self.output_constraint = ConvexPolytope(self._network_output_shape)
        elif self.network is not expression:
            raise self.translator_error("Unsupported property: Multiple networks")
        return expression

    def visit_Subscript(self, expression: Subscript):
        index = self.visit(expression.index)
        if not isinstance(index, Constant):
            raise self.translator_error(
                "Unsupported property: Symbolic subscript index"
            )
        expr = expression.expr
        if isinstance(expr, (FunctionCall, Subscript)):
            expr_indices, expr_coefs = self.visit(expr)
            coefs = expr_coefs[index.value]
            indices = expr_indices[index.value]
            return indices, coefs
        raise self.translator_error(
            "Unsupported property:"
            f" Subscript is not currently supported for {type(expr).__name__!r} expressions"
        )

    def visit_Symbol(self, expression: Symbol):
        if self.input is None:
            self.input = expression
            self.input_constraint = ConvexPolytope(
                self._network_input_shapes[expression]
            )
        elif self.input is not expression:
            raise self.translator_error("Multiple inputs detected in property")
        shape = self._network_input_shapes[expression]
        return (
            np.array([{"index": i} for i in np.ndindex(shape)]).reshape(shape),
            np.ones(shape),
        )

