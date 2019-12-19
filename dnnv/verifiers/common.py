import numpy as np

from typing import List, Optional, Type

from dnnv.nn import OperationGraph
from dnnv.nn.layers import Layer, InputLayer, FullyConnected, Convolutional
from dnnv.properties.base import (
    Constant,
    FunctionCall,
    Network,
    SlicedNetwork,
    Symbol,
    ExpressionVisitor,
)


class VerifierError(Exception):
    pass


class VerifierTranslatorError(Exception):
    pass


def as_layers(
    op_graph: OperationGraph, layer_types: Optional[List[Type[Layer]]] = None
) -> List[Layer]:
    if layer_types is None:
        layer_types = [InputLayer, FullyConnected, Convolutional]
    layers: List[Layer] = []
    while True:
        layer_match = Layer.match(op_graph, layer_types=layer_types)
        if layer_match is None:
            break
        layers.insert(0, layer_match.layer)
        op_graph = layer_match.input_op_graph
    if len(op_graph.output_operations) > 0:
        raise VerifierTranslatorError("Unsupported computation graph detected")
    return layers


class PropertyCheckResult:
    def __init__(self, name):
        self.name = name

    def __str__(self):
        return self.name

    def __invert__(self):
        if self == UNSAT:
            return SAT
        if self == SAT:
            return UNSAT
        return UNKNOWN

    def __and__(self, other):
        if not isinstance(other, PropertyCheckResult):
            return NotImplemented
        if self == UNSAT or other == UNSAT:
            return UNSAT
        if self == UNKNOWN or other == UNKNOWN:
            return UNKNOWN
        return SAT

    def __or__(self, other):
        if not isinstance(other, PropertyCheckResult):
            return NotImplemented
        if self == SAT or other == SAT:
            return SAT
        if self == UNKNOWN or other == UNKNOWN:
            return UNKNOWN
        return UNSAT


SAT = PropertyCheckResult("sat")
UNKNOWN = PropertyCheckResult("unknown")
UNSAT = PropertyCheckResult("unsat")


class PropertyExtractor(ExpressionVisitor):
    def __init__(self):
        self.op_graph = None
        self.constraint_type = None
        self.input_lower_bound = None
        self.input_upper_bound = None
        self.output_constraint = None

    def extract(self, expression):
        self.op_graph = None
        self.constraint_type = None
        self.input_lower_bound = None
        self.input_upper_bound = None
        self.output_constraint = None
        self.visit(expression)
        lb = self.input_lower_bound
        ub = self.input_upper_bound
        return self.op_graph, self.constraint_type, (lb, ub), self.output_constraint

    def assert_op_graph(self, op_graph):
        if self.op_graph is None:
            self.op_graph = op_graph
        elif self.op_graph != op_graph:
            raise VerifierTranslatorError(
                "Unsupported property type. Multiple operation graphs detected."
            )

    def assert_not_class(self, value, using="argmax"):
        if self.output_constraint is None:
            self.constraint_type = f"classification-{using}"
            self.output_constraint = {}
        elif self.constraint_type != f"classification-{using}":
            raise VerifierTranslatorError(
                f"Non-{using}-classification output constraint already defined"
            )
        if "!=" in self.output_constraint:
            raise VerifierTranslatorError(
                "More than 1 classification constraint is not currently supported"
            )
        self.output_constraint["!="] = value

    def assert_output_gte(self, value):
        if self.output_constraint is None:
            self.constraint_type = "regression"
            self.output_constraint = {}
        elif self.constraint_type != "regression":
            raise VerifierTranslatorError(
                "Non-regression output constraint already defined"
            )
        self.output_constraint[">="] = np.maximum(
            value, self.output_constraint.get(">=", value)
        )

    def assert_output_lte(self, value):
        if self.output_constraint is None:
            self.constraint_type = "regression"
            self.output_constraint = {}
        elif self.constraint_type != "regression":
            raise VerifierTranslatorError(
                "Non-regression output constraint already defined"
            )
        self.output_constraint["<="] = np.minimum(
            value, self.output_constraint.get("<=", value)
        )

    def update_input_lower_bound(self, value):
        if self.input_lower_bound is None:
            self.input_lower_bound = value
        else:
            self.input_lower_bound = np.minimum(self.input_lower_bound, value)
        if self.input_upper_bound is not None and np.any(
            self.input_lower_bound > self.input_upper_bound
        ):
            raise VerifierTranslatorError(
                "Input lower bound is greater than the upper bound"
            )

    def update_input_upper_bound(self, value):
        if self.input_upper_bound is None:
            self.input_upper_bound = value
        else:
            self.input_upper_bound = np.maximum(self.input_upper_bound, value)
        if self.input_lower_bound is not None and np.any(
            self.input_lower_bound > self.input_upper_bound
        ):
            raise VerifierTranslatorError(
                "Input upper bound is less than the lower bound"
            )

    def generic_visit(self, expression):
        raise VerifierTranslatorError("Unsupported property")

    def visit(self, expression):
        super().visit(expression)

    def visit_And(self, expression):
        super().generic_visit(expression)

    def visit_GreaterThan(self, expression):
        expr1 = expression.expr1
        expr2 = expression.expr2
        if isinstance(expr1, Symbol) and isinstance(expr2, Constant):
            self.update_input_lower_bound(expr2.value)
        elif isinstance(expr1, Constant) and isinstance(expr2, Symbol):
            self.update_input_upper_bound(expr1.value)
        else:
            raise VerifierTranslatorError("Unsupported property type")

    def visit_GreaterThanOrEqual(self, expression):
        expr1 = expression.expr1
        expr2 = expression.expr2
        if isinstance(expr1, Symbol) and isinstance(expr2, Constant):
            self.update_input_lower_bound(expr2.value)
        elif isinstance(expr1, Constant) and isinstance(expr2, Symbol):
            self.update_input_upper_bound(expr1.value)
        elif (
            isinstance(expr1, FunctionCall)
            and isinstance(expr2, Constant)
            and isinstance(expr1.function, (Network, SlicedNetwork))
        ):
            const = expr2.value
            self.assert_output_gte(const)
            self.assert_op_graph(expr1.function.concrete_value)
        elif (
            isinstance(expr1, Constant)
            and isinstance(expr2, FunctionCall)
            and isinstance(expr2.function, (Network, SlicedNetwork))
        ):
            const = expr1.value
            self.assert_output_lte(const)
            self.assert_op_graph(expr2.function.concrete_value)
        else:
            raise VerifierTranslatorError("Unsupported property type")

    def visit_LessThan(self, expression):
        expr1 = expression.expr1
        expr2 = expression.expr2
        if isinstance(expr1, Symbol) and isinstance(expr2, Constant):
            self.update_input_upper_bound(expr2.value)
        elif isinstance(expr1, Constant) and isinstance(expr2, Symbol):
            self.update_input_lower_bound(expr1.value)
        else:
            raise VerifierTranslatorError("Unsupported property type")

    def visit_LessThanOrEqual(self, expression):
        expr1 = expression.expr1
        expr2 = expression.expr2
        if isinstance(expr1, Symbol) and isinstance(expr2, Constant):
            self.update_input_upper_bound(expr2.value)
        elif isinstance(expr1, Constant) and isinstance(expr2, Symbol):
            self.update_input_lower_bound(expr1.value)
        elif (
            isinstance(expr1, FunctionCall)
            and isinstance(expr2, Constant)
            and isinstance(expr1.function, (Network, SlicedNetwork))
        ):
            const = expr2.value
            self.assert_output_lte(const)
            self.assert_op_graph(expr2.function.concrete_value)
        elif (
            isinstance(expr1, Constant)
            and isinstance(expr2, FunctionCall)
            and isinstance(expr2.function, (Network, SlicedNetwork))
        ):
            const = expr1.value
            self.assert_output_gte(const)
            self.assert_op_graph(expr2.function.concrete_value)
        else:
            raise VerifierTranslatorError("Unsupported property type")

    def visit_NotEqual(self, expression):
        expr1 = expression.expr1
        expr2 = expression.expr2
        if isinstance(expr1, FunctionCall) and isinstance(expr2, Constant):
            function_call = expr1
            const = expr2.value
        elif isinstance(expr1, Constant) and isinstance(expr2, FunctionCall):
            function_call = expr2
            const = expr1.value
        else:
            raise VerifierTranslatorError("Unsupported property type")
        args = tuple(arg for arg in function_call.args)
        kwargs = {name: value for name, value in function_call.kwargs.items()}
        if (
            function_call.function.name == "numpy.argmax"
            and len(args) == 1
            and len(kwargs) == 0
            and isinstance(args[0], FunctionCall)
            and isinstance(args[0].function, (Network, SlicedNetwork))
        ):
            self.assert_not_class(const, using="argmax")
            self.assert_op_graph(args[0].function.concrete_value)
        else:
            raise VerifierTranslatorError(
                "Unsupported function call: %s" % function_call
            )

