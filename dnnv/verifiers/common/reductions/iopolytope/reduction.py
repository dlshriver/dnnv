from __future__ import annotations

import copy
import logging
from typing import Dict, Iterator, List, Tuple, Type, Union

import numpy as np

from .....properties import *
from .....properties.visitors.inference import DetailsInference
from ..base import Property, Reduction, ReductionError
from .base import Constraint, HalfspacePolytope, HyperRectangle, Variable
from .errors import IOPolytopeReductionError
from .property import IOPolytopeProperty


class IOPolytopeReduction(Reduction):
    def __init__(
        self,
        input_constraint_type: Type[Constraint] = HyperRectangle,
        output_constraint_type: Type[Constraint] = HalfspacePolytope,
        reduction_error: Type[ReductionError] = IOPolytopeReductionError,
    ):
        super().__init__(reduction_error=reduction_error)
        self.input_constraint_type = input_constraint_type
        self.output_constraint_type = output_constraint_type
        self.logger = logging.getLogger(__name__)
        self._stack: List[Expression] = []
        self.initialize()

    def initialize(self):
        self.input = None
        self.networks = []
        self.input_constraint = self.input_constraint_type()
        self.output_constraint = self.output_constraint_type()
        self.variables: Dict[Expression, Variable] = {}
        self.indices: Dict[Expression, np.ndarray] = {}
        self.coefs: Dict[Expression, np.ndarray] = {}

    def build_property(self):
        input_lb, input_ub = self.input_constraint.as_bounds()
        if np.any(np.isinf(input_lb) | np.isinf(input_ub)):
            raise self.reduction_error(
                "Unbounded inputs detected!"
                " All inputs must have upper and lower bounds."
            )
        return IOPolytopeProperty(
            self.networks, self.input_constraint, self.output_constraint
        )

    def _reduce(self, expression: And) -> Iterator[Property]:
        self.initialize()
        if len(expression.variables) != 1:
            raise self.reduction_error(
                "At most one symbolic variable is allowed."
                f" Received: {tuple(str(v) for v in expression.variables)}"
            )
        self.visit(expression)
        prop = self.build_property()
        if not prop.input_constraint.is_consistent:
            self.logger.warning(
                "Skipping conjunction with inconsistent input constraints."
            )
            return
        if not prop.output_constraint.is_consistent:
            self.logger.warning(
                "Skipping conjunction with inconsistent output constraints."
            )
            return
        yield prop

    def reduce_property(self, expression: Expression) -> Iterator[Property]:
        if not isinstance(expression, Exists):
            raise NotImplementedError()  # TODO
        dnf_expression = expression.canonical()
        self.logger.debug("DNF: %s", dnf_expression)
        assert isinstance(dnf_expression, Or)
        DetailsInference().visit(dnf_expression)

        for conjunction in dnf_expression:
            self.logger.info("CONJUNCTION: %s", conjunction)
            yield from self._reduce(conjunction)

    def visit(self, expression):
        self._stack.append(type(expression))
        method_name = f"visit_{type(expression).__name__}"
        visitor = getattr(self, method_name, None)
        if visitor is None:
            raise self.reduction_error(
                f"Unsupported property: expression type {type(expression).__name__!r}"
                " is not currently supported"
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

    def visit_Call(self, expression: Call):
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
                    "Unsupported property: Executing networks with keyword arguments"
                    " is not currently supported"
                )
            for arg in expression.args:
                self.visit(arg)
            shape = expression.ctx.shapes[expression]
            variable = Variable(shape, str(expression.function))
            self.output_constraint.add_variable(variable)
            self.variables[expression] = variable
            self.indices[expression] = np.array(
                [i for i in np.ndindex(*shape)]
            ).reshape(*shape, len(shape))
            self.coefs[expression] = np.ones(shape)
        else:
            raise self.reduction_error(
                "Unsupported property:"
                f" Function {expression.function} is not currently supported"
            )

    def visit_Constant(self, expression: Constant):
        return

    def _add_constraint(self, expression: Union[LessThan, LessThanOrEqual]):
        if len(self._stack) > 2:
            raise self.reduction_error(
                "Not Canonical: "
                f"{type(expression).__name__!r} expression below expected level"
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
                "Invalid property:"
                " Adding expressions with different shapes is not supported"
            )
        c_shape = c_shapes.pop()
        if rhs.shape != c_shape:
            rhs = np.zeros(c_shape) + rhs
        if rhs.shape != c_shape:
            raise self.reduction_error(
                "Invalid property:"
                " Comparing expressions with different shapes is not supported"
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
                    constraints.append(([(var, idx)], [coef], rhs.item()))
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
                        "Invalid property:"
                        " Adding expressions with different shapes is not supported"
                    )
                c_shape = shape
                num_constraints = np.product(shape)
                if len(constraints) == 0:
                    constraints = [([], [], rhs[i]) for i in np.ndindex(*rhs.shape)]
                elif len(constraints) == 1:
                    constraints = [
                        (
                            copy.deepcopy(constraints[0][0]),
                            copy.deepcopy(constraints[0][1]),
                            rhs[i],
                        )
                        for i in np.ndindex(*rhs.shape)
                    ]
                if not len(constraints) == num_constraints:
                    raise self.reduction_error(
                        "Invalid property:"
                        " Adding expressions with different shapes is not supported"
                    )
                for c, idx_, coef_ in zip(
                    constraints,
                    idx.reshape((num_constraints, -1)),
                    coef.reshape(num_constraints),
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
                    "Networks with multiple output operations"
                    " are not currently supported"
                )
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
            if expression not in expression.ctx.shapes:
                raise self.reduction_error(f"Unknown shape for variable {expression}")
            variable = Variable(expression.ctx.shapes[expression], str(expression))
            self.input_constraint.add_variable(variable)
        elif self.input is not expression:
            raise self.reduction_error("Multiple inputs detected in property")
        shape = expression.ctx.shapes[expression]
        self.variables[expression] = Variable(
            expression.ctx.shapes[expression], str(expression)
        )
        self.indices[expression] = np.array(list(np.ndindex(*shape))).reshape(
            *shape, len(shape)
        )
        self.coefs[expression] = np.ones(shape)


__all__ = ["IOPolytopeReduction"]
