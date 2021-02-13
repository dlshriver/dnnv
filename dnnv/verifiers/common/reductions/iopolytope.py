import copy
import logging
import numpy as np

from abc import ABC, abstractmethod
from collections import namedtuple
from scipy.optimize import linprog
from typing import Dict, Generator, List, Optional, Tuple, Type, Union

from dnnv.nn import OperationGraph, OperationTransformer
from dnnv.properties import ExpressionVisitor
from dnnv.properties.base import (
    Add,
    And,
    Constant,
    Expression,
    Exists,
    FunctionCall,
    LessThan,
    LessThanOrEqual,
    Multiply,
    Network,
    Or,
    Subscript,
    Symbol,
)

from .base import Property, Reduction
from ..errors import VerifierTranslatorError


class Variable:
    _count = 0

    def __init__(self, shape: Tuple[int, ...], name: Optional[str] = None):
        self.shape = shape
        self.name = name
        if self.name is None:
            self.name = f"x_{Variable._count}"
        Variable._count += 1

    def size(self):
        return np.product(self.shape)

    def __str__(self):
        return self.name

    def __repr__(self):
        return f"Variable({self.shape}, {self.name!r})"

    def __hash__(self):
        return hash(self.shape) * hash(self.name)

    def __eq__(self, other):
        if not isinstance(other, Variable):
            return False
        return (self.name == self.name) and (self.shape == self.shape)


class Constraint(ABC):
    def __init__(self, variable: Variable):
        self.variables = {variable: 0}

    @property
    def is_consistent(self):
        return None

    @property
    def num_variables(self):
        return len(self.variables)

    def size(self):
        return sum(variable.size() for variable in self.variables)

    def add_variable(self, variable: Variable):
        if variable not in self.variables:
            self.variables[variable] = self.size()
        return self

    def unravel_index(self, index: int):
        for variable, size in sorted(self.variables.items(), key=lambda kv: -kv[1]):
            if index >= size:
                return variable, np.unravel_index(index - size, variable.shape)
        raise ValueError(
            f"index {index} is out of bounds for constraint with size {self.size()}"
        )

    @abstractmethod
    def update_constraint(self, variables, indices, coefficients, b, is_open=False):
        pass

    @abstractmethod
    def validate(self, *x):
        pass


Halfspace = namedtuple("Halfspace", ["indices", "coefficients", "b", "is_open"])


class HalfspacePolytope(Constraint):
    def __init__(self, variable):
        super().__init__(variable)
        self.halfspaces: List[Halfspace] = []
        self._A = None
        self._b = None
        self._lower_bound = np.zeros(self.size()) - np.inf
        self._upper_bound = np.zeros(self.size()) + np.inf

    def as_matrix_inequality(self, tol=None):
        assert tol is None
        return self._A, self._b

    def as_bounds(self, tol=None):
        return self._lower_bound, self._upper_bound

    @property
    def is_consistent(self):
        A, b = self.as_matrix_inequality()
        obj = np.zeros(A.shape[1])
        try:
            result = linprog(obj, A_ub=A, b_ub=b, bounds=(None, None),)
        except ValueError as e:
            if "The problem is (trivially) infeasible" in e.args[0]:
                return False
            raise e
        if result.status == 4:
            return None
        elif result.status == 2:  # infeasible
            return False
        elif result.status == 0:  # feasible
            return True
        return None  # unknown

    def _update_bounds(self, indices, coefficients, b, is_open=False):
        n = self.size()
        if self._lower_bound is None:
            self._lower_bound = np.zeros((n,)) - np.inf
        if self._upper_bound is None:
            self._upper_bound = np.zeros((n,)) + np.inf

        if len(indices) == 1:
            index = indices[0]
            coef_sign = np.sign(coefficients[0])
            value = b / coefficients[0]
            if coef_sign < 0:
                if is_open:
                    value = np.nextafter(value, value + 1)
                self._lower_bound[index] = max(value, self._lower_bound[index])
            elif coef_sign > 0:
                if is_open:
                    value = np.nextafter(value, value - 1)
                self._upper_bound[index] = min(value, self._upper_bound[index])
        else:
            for i in indices:
                obj = np.zeros(n)
                try:
                    obj[i] = 1
                    result = linprog(
                        obj, A_ub=self._A, b_ub=self._b, bounds=(None, None)
                    )
                    if result.status == 0:
                        self._lower_bound[i] = result.x[i]
                except ValueError as e:
                    if (
                        e.args[0]
                        != "The algorithm terminated successfully and determined that the problem is infeasible."
                    ):
                        raise e
                try:
                    obj[i] = -1
                    result = linprog(
                        obj, A_ub=self._A, b_ub=self._b, bounds=(None, None)
                    )
                    if result.status == 0:
                        self._upper_bound[i] = result.x[i]
                except ValueError as e:
                    if (
                        e.args[0]
                        != "The algorithm terminated successfully and determined that the problem is infeasible."
                    ):
                        raise e

    def update_constraint(self, variables, indices, coefficients, b, is_open=False):
        flat_indices = [
            self.variables[var] + np.ravel_multi_index(idx, var.shape)
            for var, idx in zip(variables, indices)
        ]
        halfspace = Halfspace(flat_indices, coefficients, b, is_open)
        self.halfspaces.append(halfspace)

        n = self.size()
        _A = np.zeros((1, n))
        _b = np.zeros((1,))
        for i, a in zip(flat_indices, coefficients):
            _A[0, i] = a
        _b[0] = b
        if is_open:
            _b[0] = np.nextafter(b, b - 1)
        if self._A is None:
            self._A = _A
            self._b = _b
        else:
            self._A = np.vstack([self._A, _A])
            self._b = np.hstack([self._b, _b])

        self._update_bounds(flat_indices, coefficients, b, is_open=is_open)

    def validate(self, *x, threshold=1e-6):
        x_flat = np.concatenate([x_.flatten() for x_ in x])
        for hs in self.halfspaces:
            t = sum(c * x_flat[i] for c, i in zip(hs.coefficients, hs.indices))
            b = hs.b
            if hs.is_open:
                b = np.nextafter(b, b - 1)
            if (t - b) > threshold:
                return False
        return True

    def __str__(self):
        strs = []
        for hs in self.halfspaces:
            lhs_strs = []
            for i, c in zip(hs.indices, hs.coefficients):
                variable, index = self.unravel_index(i)
                lhs_strs.append(f"{c} * {variable}[{index}]")
            if hs.is_open:
                strs.append(" + ".join(lhs_strs) + f" < {hs.b}")
            else:
                strs.append(" + ".join(lhs_strs) + f" <= {hs.b}")
        return "\n".join(strs)


class HyperRectangle(HalfspacePolytope):
    @property
    def is_consistent(self):
        if (self._lower_bound > self._upper_bound).any():
            return False
        return True

    @property
    def lower_bounds(self):
        lbs = []
        for variable, start_index in self.variables.items():
            size = variable.size()
            lbs.append(
                self._lower_bound[start_index : start_index + size].reshape(
                    variable.shape
                )
            )
        return lbs

    @property
    def upper_bounds(self):
        ubs = []
        for variable, start_index in self.variables.items():
            size = variable.size()
            ubs.append(
                self._upper_bound[start_index : start_index + size].reshape(
                    variable.shape
                )
            )
        return ubs

    def add_variable(self, variable):
        super().add_variable(variable)
        size = variable.size()
        self._lower_bound = np.concatenate([self._lower_bound, np.zeros(size) - np.inf])
        self._upper_bound = np.concatenate([self._upper_bound, np.zeros(size) + np.inf])
        return self

    def _update_bounds(self, *args, **kwargs):
        pass

    def update_constraint(self, variables, indices, coefficients, b, is_open=False):
        if len(indices) > 1:
            raise ValueError(
                "HyperRectangle constraints can only constrain a single dimension"
            )
        flat_index = self.variables[variables[0]] + np.ravel_multi_index(
            indices[0], variables[0].shape
        )
        coef = np.sign(coefficients[0])
        value = b / coefficients[0]
        if coef < 0:
            if is_open:
                value = np.nextafter(value, value + 1)
            self._lower_bound[flat_index] = max(value, self._lower_bound[flat_index])
        elif coef > 0:
            if is_open:
                value = np.nextafter(value, value - 1)
            self._upper_bound[flat_index] = min(value, self._upper_bound[flat_index])
        super().update_constraint(variables, indices, coefficients, b, is_open)

    def __str__(self):
        strs = []
        for i in range(self.size()):
            lb = self._lower_bound[i]
            ub = self._upper_bound[i]
            variable, index = self.unravel_index(i)
            strs.append(f"{lb:f} <= {variable}[{index}] <= {ub:f}")
        return "\n".join(strs)


class IOPolytopeProperty(Property):
    def __init__(
        self,
        networks: List[Network],
        input_constraint: Constraint,
        output_constraint: Constraint,
    ):
        self.networks = networks
        self.input_constraint = input_constraint
        setattr(self.input_constraint, "_varname", "x")
        self.output_constraint = output_constraint
        setattr(
            self.output_constraint,
            "_varname",
            [f"{network}(x)" for network in self.networks],
        )
        # TODO : move Merger out of this function
        class Merger(OperationTransformer):
            # TODO : merge common layers (e.g. same normalization, reshaping of input)
            def __init__(self):
                self.output_operations = []
                self.input_operations = {}

            def merge(self, operation_graphs: List[OperationGraph]):
                for op_graph in operation_graphs:
                    for op in op_graph.output_operations:
                        self.output_operations.append(self.visit(op))
                return OperationGraph(self.output_operations)

            def visit_Input(self, operation):
                input_details = (operation.dtype, tuple(operation.shape))
                if input_details not in self.input_operations:
                    self.input_operations[input_details] = self.generic_visit(operation)
                return self.input_operations[input_details]

        self.op_graph = Merger().merge([n.value for n in self.networks])

    def __str__(self):
        strs = ["Property:"]
        strs += ["  Networks:"] + ["    " + str(self.networks)]
        strs += ["  Input Constraint:"] + [
            "    " + s for s in str(self.input_constraint).split("\n")
        ]
        strs += ["  Output Constraint:"] + [
            "    " + s for s in str(self.output_constraint).split("\n")
        ]
        return "\n".join(strs)

    def validate_counter_example(self, cex, threshold=1e-6):
        if not self.input_constraint.validate(cex, threshold=threshold):
            return False, "Invalid counter example found: input outside bounds."
        output = self.op_graph(cex)
        if not self.output_constraint.validate(output, threshold=threshold):
            return False, "Invalid counter example found: output outside bounds."
        return True, None

    def prefixed_and_suffixed_op_graph(
        self,
    ) -> Tuple[OperationGraph, Tuple[List[np.ndarray], List[np.ndarray]]]:
        import dnnv.nn.operations as operations

        if not isinstance(self.input_constraint, HyperRectangle):
            raise ValueError(
                f"{type(self.input_constraint).__name__} input constraints are not yet supported"
            )

        suffixed_op_graph = self.suffixed_op_graph()

        class PrefixTransformer(OperationTransformer):
            def __init__(self, lbs, ubs):
                super().__init__()
                self.lbs = lbs
                self.ubs = ubs
                self._input_count = 0
                self.final_lbs = []
                self.final_ubs = []

            def visit_Input(
                self, operation: operations.Input
            ) -> Union[operations.Conv, operations.Gemm]:
                new_op: Union[operations.Conv, operations.Gemm]
                input_shape = self.lbs[self._input_count].shape
                if len(input_shape) == 2:
                    ranges = self.ubs[self._input_count] - self.lbs[self._input_count]
                    mins = self.lbs[self._input_count]
                    new_op = operations.Gemm(
                        operation, np.diag(ranges.flatten()), mins.flatten()
                    )
                    self.final_lbs.append(np.zeros(mins.shape))
                    self.final_ubs.append(np.ones(mins.shape))
                elif len(input_shape) == 4:
                    b = self.lbs[self._input_count].min(axis=(0, 2, 3))
                    r = self.ubs[self._input_count].max(axis=(0, 2, 3)) - b
                    c = len(r)
                    w = np.zeros((c, c, 1, 1))
                    for i in range(c):
                        w[i, i, 0, 0] = 1
                    w = w * r.reshape((c, 1, 1, 1))
                    new_op = operations.Conv(operation, w, b)
                    test_op_graph = OperationGraph([new_op])
                    tile_shape = list(self.lbs[self._input_count].shape)
                    tile_shape[1] = 1
                    tiled_b = np.tile(b.reshape((1, -1, 1, 1)), tile_shape)
                    tiled_r = np.tile(r.reshape((1, -1, 1, 1)), tile_shape)
                    self.final_lbs.append(
                        (self.lbs[self._input_count] - tiled_b) / tiled_r
                    )
                    self.final_ubs.append(
                        (self.ubs[self._input_count] - tiled_b) / tiled_r
                    )
                else:
                    raise NotImplementedError(
                        f"Cannot prefix network with input shape {input_shape}"
                    )
                self._input_count += 1
                return new_op

        prefix_transformer = PrefixTransformer(
            self.input_constraint.lower_bounds, self.input_constraint.upper_bounds,
        )
        prefixed_op_graph = OperationGraph(suffixed_op_graph.walk(prefix_transformer))
        return (
            prefixed_op_graph.simplify(),
            (prefix_transformer.final_lbs, prefix_transformer.final_ubs),
        )

    def suffixed_op_graph(self) -> OperationGraph:
        import dnnv.nn.operations as operations

        if not isinstance(self.output_constraint, HalfspacePolytope):
            raise ValueError(
                f"{type(self.output_constraint).__name__} output constraints are not yet supported"
            )
        if len(self.op_graph.output_operations) == 1:
            new_output_op = self.op_graph.output_operations[0]
        else:
            output_operations = [
                operations.Flatten(o) for o in self.op_graph.output_operations
            ]
            new_output_op = operations.Concat(output_operations, axis=1)
        size = self.output_constraint.size()
        k = len(self.output_constraint.halfspaces)
        W = np.zeros((size, k), dtype=np.float32)
        b = np.zeros(k, dtype=np.float32)
        for n, hs in enumerate(self.output_constraint.halfspaces):
            b[n] = -hs.b
            if hs.is_open:
                b[n] += 1e-6  # TODO : remove magic number
            for i, c in zip(hs.indices, hs.coefficients):
                W[i, n] = c
        new_output_op = operations.Add(operations.MatMul(new_output_op, W), b)
        new_output_op = operations.Relu(new_output_op)

        W_mask = np.zeros((k, 1), dtype=np.float32)
        b_mask = np.zeros(1, dtype=np.float32)
        for i in range(k):
            W_mask[i, 0] = 1
        new_output_op = operations.Add(operations.MatMul(new_output_op, W_mask), b_mask)
        return OperationGraph([new_output_op]).simplify()


class IOPolytopeReduction(Reduction):
    def __init__(
        self,
        input_constraint_type: Type[Constraint] = HyperRectangle,
        output_constraint_type: Type[Constraint] = HalfspacePolytope,
        reduction_error: Type[VerifierTranslatorError] = VerifierTranslatorError,
    ):
        super().__init__(reduction_error=reduction_error)
        self.input_constraint_type = input_constraint_type
        self.output_constraint_type = output_constraint_type
        self.logger = logging.getLogger(__name__)
        self._stack: List[Expression] = []
        self._network_input_shapes: Dict[Expression, Tuple[int, ...]] = {}
        self._network_output_shapes: Dict[Network, Tuple[int, ...]] = {}
        self.initialize()

    def initialize(self):
        self.input = None
        self.networks = []
        self.input_constraint = None
        self.output_constraint = None
        self.variables: Dict[Expression, Variable] = {}
        self.indices: Dict[Expression, np.ndarray] = {}
        self.coefs: Dict[Expression, np.ndarray] = {}

    def build_property(self):
        return IOPolytopeProperty(
            self.networks, self.input_constraint, self.output_constraint
        )

    def _reduce(self, expression: And) -> Generator[Property, None, None]:
        self.initialize()
        if len(expression.variables) != 1:
            raise self.reduction_error("Exactly one network input is required")
        self.visit(expression)
        prop = self.build_property()
        if prop.input_constraint.is_consistent == False:
            self.logger.warning(
                "Skipping conjunction with inconsistent input constraints."
            )
            return
        if prop.output_constraint.is_consistent == False:
            self.logger.warning(
                "Skipping conjunction with inconsistent output constraints."
            )
            return
        yield prop

    def reduce_property(
        self, expression: Expression
    ) -> Generator[Property, None, None]:
        if not isinstance(expression, Exists):
            raise NotImplementedError()  # TODO
        dnf_expression = Or(~(~expression).canonical())
        self.logger.debug("DNF: %s", dnf_expression)

        for conjunction in dnf_expression:
            self.logger.info("CONJUNCTION: %s", conjunction)
            yield from self._reduce(conjunction)

    def visit(self, expression):
        self._stack.append(type(expression))
        method_name = "visit_%s" % expression.__class__.__name__
        visitor = getattr(self, method_name, None)
        if visitor is None:
            raise self.reduction_error(
                "Unsupported property:"
                f" expression type {type(expression).__name__!r} is not currently supported"
            )
        result = visitor(expression)
        self._stack.pop()
        return result

    def visit_Add(self, expression: Add):
        if len(self._stack) > 3:
            raise self.reduction_error(
                "Not Canonical:"
                f" {type(expression).__name__!r} expression found below expected level"
            )
        indices = {}
        coefs = {}
        for expr in expression.expressions:
            self.visit(expr)
            indices[expr] = self.indices[expr]
            coefs[expr] = self.coefs[expr]
        self.indices = indices
        self.coefs = coefs

    def visit_And(self, expression: And):
        if len(self._stack) != 1:
            raise self.reduction_error(
                "Not Canonical: 'And' expression not at top level"
            )
        for expr in sorted(expression.expressions, key=lambda e: -len(e.networks)):
            self.visit(expr)

    def visit_Constant(self, expression: Constant):
        return

    def visit_FunctionCall(self, expression: FunctionCall):
        if isinstance(expression.function, Network):
            self.visit(expression.function)
            input_details = expression.function.value.input_details
            if len(expression.args) != len(input_details):
                raise self.reduction_error(
                    "Invalid property:"
                    f" Not enough inputs for network '{expression.function}'"
                )
            if len(expression.kwargs) > 0:
                raise self.reduction_error(
                    "Unsupported property:"
                    f" Executing networks with keyword arguments is not currently supported"
                )
            for arg, d in zip(expression.args, input_details):
                if arg in self._network_input_shapes:
                    if any(
                        i1 != i2 and i2 > 0
                        for i1, i2 in zip(
                            self._network_input_shapes[arg], tuple(d.shape)
                        )
                    ):
                        raise self.reduction_error(
                            f"Invalid property: variable with multiple shapes: '{arg}'"
                        )
                self._network_input_shapes[arg] = tuple(
                    i if i > 0 else 1 for i in d.shape
                )
                self.visit(arg)
            shape = self._network_output_shapes[expression.function]
            self.variables[expression] = self.variables[expression.function]
            self.indices[expression] = np.array([i for i in np.ndindex(shape)]).reshape(
                shape + (len(shape),)
            )
            self.coefs[expression] = np.ones(shape)
        else:
            raise self.reduction_error(
                "Unsupported property:"
                f" Function {expression.function} is not currently supported"
            )

    def _add_constraint(self, expression: Union[LessThan, LessThanOrEqual]):
        if len(self._stack) > 2:
            raise self.reduction_error(
                f"Not Canonical: {type(expression).__name__!r} expression below expected level"
            )
        if not isinstance(expression.expr1, Add):
            raise self.reduction_error(
                "Not Canonical:"
                f" LHS of {type(expression).__name__!r} is not an 'Add' expression"
            )
        if not isinstance(expression.expr2, Constant):
            raise self.reduction_error(
                "Not Canonical:"
                f" RHS of {type(expression).__name__!r} is not a 'Constant' expression"
            )
        self.indices.clear()
        self.coefs.clear()
        self.visit(expression.expr1)
        rhs = np.asarray(expression.expr2.value)
        is_open = False
        if isinstance(expression, LessThan):
            is_open = True

        def zip_dict_items(*dicts):
            for i in set(dicts[0]).intersection(*dicts[1:]):
                yield (i,) + tuple(d[i] for d in dicts)

        c_shapes = set(c.shape for c in self.coefs.values())
        if len(c_shapes) > 1:
            raise self.reduction_error(
                "Invalid property: Adding expressions with different shapes is not supported"
            )
        c_shape = c_shapes.pop()
        if rhs.shape != c_shape:
            rhs = np.zeros(c_shape) + rhs
        if rhs.shape != c_shape:
            raise self.reduction_error(
                "Invalid property: Comparing expressions with different shapes is not supported"
            )

        constraints: List[
            Tuple[List[Tuple[Variable, Tuple[int, ...]]], List[float], float]
        ] = []
        for key, var, idx, coef in zip_dict_items(
            self.variables, self.indices, self.coefs
        ):
            if len(idx.shape) == 1:
                idx = tuple(idx)
                if len(constraints) == 0:
                    constraints.append(([(var, idx)], [coef], rhs))
                else:
                    for c in constraints:
                        var_idx = (var, idx)
                        if var_idx in c[0]:
                            c[1][c[0].index(var_idx)] += coef
                        else:
                            c[0].append(var_idx)
                            c[1].append(coef)
            else:
                shape = coef.shape
                if c_shape is not None and c_shape != shape:
                    raise self.reduction_error(
                        "Invalid property: Adding expressions with different shapes is not supported"
                    )
                c_shape = shape
                new_shape = np.product(shape), len(shape)
                if len(constraints) == 0:
                    constraints = [([], [], rhs[i]) for i in np.ndindex(rhs.shape)]
                elif len(constraints) == 1:
                    constraints = [
                        (
                            copy.deepcopy(constraints[0][0]),
                            copy.deepcopy(constraints[0][1]),
                            rhs[i],
                        )
                        for i in np.ndindex(rhs.shape)
                    ]
                if not len(constraints) == new_shape[0]:
                    raise self.reduction_error(
                        "Invalid property: Adding expressions with different shapes is not supported"
                    )
                for c, idx_, coef_ in zip(
                    constraints, idx.reshape(new_shape), coef.reshape(new_shape[0])
                ):
                    var_idx = (var, tuple(idx_))
                    if var_idx in c[0]:
                        c[1][c[0].index(var_idx)] += coef_
                    else:
                        c[0].append(var_idx)
                        c[1].append(coef_)

        if len(expression.networks) == 0:
            current_constraint = self.input_constraint
        else:
            current_constraint = self.output_constraint
        for c in constraints:
            variables, indices = zip(*c[0])
            current_constraint.update_constraint(
                variables, indices, c[1], c[2], is_open=is_open
            )

    def visit_LessThan(self, expression: LessThan):
        self._add_constraint(expression)

    def visit_LessThanOrEqual(self, expression: LessThanOrEqual):
        self._add_constraint(expression)

    def visit_Multiply(self, expression: Multiply):
        constants = []
        symbols = []
        for expr in expression.expressions:
            self.visit(expr)
            if expr.is_concrete:
                constants.append(expr.value)
            else:
                symbols.append(expr)
        if len(symbols) > 1:
            raise self.reduction_error(
                "Unsupported property: Multiplication of symbolic values"
            )
        self.variables[expression] = self.variables[symbols[0]]
        self.indices[expression] = self.indices[symbols[0]]
        self.coefs[expression] = np.product(constants) * self.coefs[symbols[0]]

    def visit_Network(self, expression: Network):
        # TODO : handle networks with multiple outputs
        # TODO : handle networks with multiple inputs
        if expression not in self.networks:
            self.networks.append(expression)
            if len(expression.value.output_operations) > 1:
                raise NotImplementedError(
                    "Networks with multiple output operations are not currently supported"
                )
            if expression not in self._network_output_shapes:
                self._network_output_shapes[expression] = expression.value.output_shape[
                    0
                ]
            elif (
                self._network_output_shapes[expression]
                != expression.value.output_shape[0]
            ):
                raise self.reduction_error(
                    f"Invalid property: network with multiple shapes: '{expression}'"
                )
            variable = Variable(
                self._network_output_shapes[expression], str(expression)
            )
            if self.output_constraint is None:
                self.output_constraint = self.output_constraint_type(variable)
            else:
                self.output_constraint = self.output_constraint.add_variable(variable)
        variable = Variable(self._network_output_shapes[expression], str(expression))
        self.variables[expression] = variable
        return expression

    def visit_Subscript(self, expression: Subscript):
        if not isinstance(expression.index, Constant):
            raise self.reduction_error("Unsupported property: Symbolic subscript index")
        index = expression.index.value
        self.visit(expression.expr)
        self.variables[expression] = self.variables[expression.expr]
        self.indices[expression] = self.indices[expression.expr][index]
        self.coefs[expression] = self.coefs[expression.expr][index]

    def visit_Symbol(self, expression: Symbol):
        if self.input is None:
            self.input = expression
            if expression not in self._network_input_shapes:
                raise self.reduction_error(f"Unknown shape for variable {expression}")
            variable = Variable(self._network_input_shapes[expression], str(expression))
            self.input_constraint = self.input_constraint_type(variable)
        elif self.input is not expression:
            raise self.reduction_error("Multiple inputs detected in property")
        shape = self._network_input_shapes[expression]
        self.variables[expression] = Variable(
            self._network_input_shapes[expression], str(expression)
        )
        self.indices[expression] = np.array([i for i in np.ndindex(shape)]).reshape(
            shape + (len(shape),)
        )
        self.coefs[expression] = np.ones(shape)


__all__ = [
    "HalfspacePolytope",
    "HyperRectangle",
    "IOPolytopeProperty",
    "IOPolytopeReduction",
]
