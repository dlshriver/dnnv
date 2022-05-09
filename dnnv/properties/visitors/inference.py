from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np

from ...errors import DNNVError
from ...nn import OperationGraph
from .. import Constant, Context, Expression, expressions, get_context
from .base import ExpressionVisitor


class DNNVInferenceError(DNNVError):
    pass


class DNNVShapeError(DNNVInferenceError):
    pass


class DNNVTypeError(DNNVInferenceError):
    pass


def _check_assertions(assertions: Sequence[Expression]) -> bool:
    from dnnv.properties.transformers import PropagateConstants

    transformer = PropagateConstants()
    transformer._top_level = False

    expr = expressions.And(*assertions)
    if expr.is_concrete:
        return expr.value
    result = transformer.visit(expr)
    if result.is_concrete and not result.value:
        return False
    return True


def _broadcastable(shape1, shape2):
    try:
        _ = np.broadcast_shapes(shape1, shape2)
    except:
        return False
    return True


def _get_type(value):
    if isinstance(value, np.ndarray):
        return value.dtype
    if isinstance(value, (int, float)):
        return np.min_scalar_type(value)
    return type(value)


def resolve_Equal(assertions: Sequence[expressions.Expression]):
    concretized = False
    for assertion in assertions:
        if assertion.is_concrete:
            continue
        if isinstance(assertion, expressions.Equal):
            if (
                isinstance(assertion.expr1, expressions.Symbol)
                and assertion.expr2.is_concrete
            ):
                assertion.expr1.concretize(assertion.expr2.value)
                concretized = True
            elif (
                isinstance(assertion.expr2, expressions.Symbol)
                and assertion.expr1.is_concrete
            ):
                assertion.expr2.concretize(assertion.expr1.value)
                concretized = True
    if concretized:
        resolve_Equal(assertions)


class DetailsInference(ExpressionVisitor):
    def __init__(self):
        super().__init__()
        self.shapes: Dict[Expression, Expression] = {}
        self.types: Dict[Expression, Expression] = {}
        self.assertions: List[expressions.LogicalExpression] = []
        self.inference_ctx = Context()
        self.base_ctx: Optional[Context] = None
        self._symbol_count = 0

    def generic_visit(
        self, expression: expressions.Expression
    ) -> expressions.Expression:
        raise NotImplementedError(
            f"DetailsInference is not yet implemented for expression type: {type(expression).__name__}"
        )

    def visit(self, expression):
        if not isinstance(expression, Expression):
            expression = Constant(expression, ctx=self.base_ctx or get_context())
        if self._top_level:
            self.inference_ctx.__enter__()
            self.base_ctx = ctx = expression.ctx
            for expr, shape in ctx.shapes.items():
                self.shapes[expr] = Constant(shape)
            for expr, dtype in ctx.types.items():
                self.types[expr] = Constant(dtype)
        assert expression.ctx is self.base_ctx
        super().visit(expression)
        resolve_Equal(self.assertions)
        if self._top_level:
            self.inference_ctx.__exit__()
            for k in self.shapes:
                if k in ctx.shapes:
                    assert (
                        ctx.shapes[k] == self.shapes[k].value
                    ), f"{ctx.shapes[k]} != {self.shapes[k].value}"
                    assert (
                        ctx.types[k] == self.types[k]
                    ), f"{ctx.types[k]} != {self.types[k]}"
                else:
                    if self.shapes[k].is_concrete:
                        shape = self.shapes[k].value
                        assert isinstance(shape, tuple)
                        ctx.shapes[k] = shape
                    if self.types[k].is_concrete:
                        ctx.types[k] = self.types[k].value

    def get_symbolic_detail(self, detail_type: str) -> expressions.Symbol:
        self._symbol_count += 1
        return expressions.Symbol(
            f"!{detail_type}!{self._symbol_count}", ctx=self.inference_ctx
        )

    def set_details(
        self,
        expression,
        shape: Optional[Expression] = None,
        dtype: Optional[Expression] = None,
    ):
        if expression not in self.shapes:
            if shape is None:
                shape = self.get_symbolic_detail("shape")
            self.shapes[expression] = shape
        elif shape is not None:
            current_shape = self.shapes[expression]
            if (
                isinstance(current_shape, expressions.Symbol)
                and not current_shape.is_concrete
            ):
                current_shape.concretize(shape)
            elif shape.value != current_shape.value:
                raise DNNVShapeError(
                    f"Multiple shapes for expression {expression!r}: {shape} and {current_shape}"
                )
        if expression not in self.types:
            if dtype is None:
                dtype = self.get_symbolic_detail("dtype")
            self.types[expression] = dtype
        elif dtype is not None:
            current_dtype = self.types[expression]
            if (
                isinstance(current_dtype, expressions.Symbol)
                and not current_dtype.is_concrete
            ):
                current_dtype.concretize(dtype)
            elif dtype.value != current_dtype.value:
                raise DNNVTypeError(
                    f"Multiple types for expression {expression!r}: {dtype} and {current_dtype}"
                )

    def visit_Add(self, expression: expressions.Add):
        dtype = None
        shape = None
        shape_assertions: List[expressions.LogicalExpression] = []
        for expr in expression:
            self.visit(expr)
            if shape is None:
                shape = self.shapes[expr]
            elif shape.is_concrete and self.shapes[expr].is_concrete:
                shape_ = shape.value
                expr_shape = self.shapes[expr].value
                if not _broadcastable(shape_, expr_shape):
                    raise DNNVShapeError(
                        f"Incompatible shapes in Add: {shape_} and {expr_shape}"
                    )
                shape_assertions.append(Constant(_broadcastable(shape_, expr_shape)))
                shape = Constant(np.broadcast_shapes(shape_, expr_shape))
            else:
                shape_assertions.append(
                    Constant(_broadcastable)(shape, self.shapes[expr])
                )
                shape = Constant(np.broadcast_shapes)(shape, self.shapes[expr])
            if dtype is None:
                dtype = self.types[expr]
            elif dtype.is_concrete and self.types[expr].is_concrete:
                dtype_ = dtype.value
                expr_dtype = self.types[expr].value
                dtype = Constant(np.result_type(dtype_, expr_dtype))
            else:
                dtype = Constant(np.result_type)(dtype, self.types[expr])

        if not _check_assertions(shape_assertions):
            raise DNNVShapeError(f"Incompatible shapes in Add")
        self.assertions.extend(shape_assertions)
        # TODO : check type assertions

        self.set_details(expression, shape=shape, dtype=dtype)

    def visit_And(self, expression: expressions.And):
        shape = None
        shape_assertions: List[expressions.LogicalExpression] = []
        type_assertions: List[expressions.LogicalExpression] = []
        for expr in expression:
            self.visit(expr)
            if shape is None:
                shape = self.shapes[expr]
            elif shape.is_concrete and self.shapes[expr].is_concrete:
                shape_ = shape.value
                expr_shape = self.shapes[expr].value
                if not _broadcastable(shape_, expr_shape):
                    raise DNNVShapeError(
                        f"Incompatible shapes in And: {shape_} and {expr_shape}"
                    )
                shape_assertions.append(Constant(_broadcastable(shape_, expr_shape)))
                shape = Constant(np.broadcast_shapes(shape_, expr_shape))
            else:
                shape_assertions.append(
                    Constant(_broadcastable)(shape, self.shapes[expr])
                )
                shape = Constant(np.broadcast_shapes)(shape, self.shapes[expr])
            type_assertions.append(self.types[expr] == Constant(bool))

        if not _check_assertions(shape_assertions):
            raise DNNVShapeError(f"Incompatible shapes in And")
        self.assertions.extend(shape_assertions)
        if not _check_assertions(type_assertions):
            raise DNNVTypeError(f"Incompatible types in And")
        self.assertions.extend(type_assertions)

        self.set_details(expression, shape=shape, dtype=Constant(bool))

    def visit_Attribute(self, expression: expressions.Attribute):
        self.visit(expression.expr1)
        self.visit(expression.expr2)

        shape = Constant(np.shape)(expression)
        dtype = Constant(_get_type)(expression)

        self.set_details(expression, shape=shape, dtype=dtype)

    def visit_Call(self, expression: expressions.Call):
        self.visit(expression.function)
        for arg in expression.args:
            self.visit(arg)
        for kwarg_value in expression.kwargs.values():
            self.visit(kwarg_value)

        shape: Optional[Expression] = None
        dtype: Optional[Expression] = None
        shape_assertions = []
        type_assertions = []
        if (
            self.types[expression.function].is_concrete
            and isinstance(expression.function, expressions.Network)
            and expression.function.is_concrete
        ):
            assert len(expression.kwargs) == 0
            op_graph: OperationGraph = expression.function.value
            input_details = op_graph.input_details
            assert len(expression.args) == len(input_details)
            for arg, details in zip(expression.args, input_details):
                shape_assertions.append(
                    self.shapes[arg] == Constant(tuple(abs(s) for s in details.shape))
                )
                type_assertions.append(self.types[arg] == Constant(details.dtype))
            output_details = op_graph.output_details
            if len(output_details) == 1:
                shape = Constant(output_details[0].shape)
                dtype = Constant(output_details[0].dtype)
            else:
                raise DNNVInferenceError(
                    "OperationGraph inference with multiple output operations is not currently supported."
                )
        else:
            empty = Constant(np.empty)
            func_out = expression.function(
                *(
                    arg if arg.is_concrete else empty(self.shapes[arg], self.types[arg])
                    for arg in expression.args
                ),
                **{
                    kw: arg
                    if arg.is_concrete
                    else empty(self.shapes[arg], self.types[arg])
                    for kw, arg in expression.kwargs.items()
                },
            )
            shape = Constant(np.shape)(func_out)
            dtype = Constant(_get_type)(func_out)

        if not _check_assertions(shape_assertions):
            raise DNNVShapeError(f"Incompatible shapes in Call")
        self.assertions.extend(shape_assertions)
        if not _check_assertions(type_assertions):
            raise DNNVTypeError(f"Incompatible types in Call")
        self.assertions.extend(type_assertions)

        self.set_details(expression, shape=shape, dtype=dtype)

    def visit_Constant(self, expression: expressions.Constant):
        value = expression.value
        if isinstance(value, np.ndarray):
            shape = Constant(value.shape)
            dtype = Constant(value.dtype)
        elif isinstance(value, (list, tuple)):
            arr = np.asarray(value)
            shape = Constant(arr.shape)
            dtype = Constant(arr.dtype)
        elif isinstance(value, (int, float)):
            shape = Constant(())
            dtype = Constant(np.min_scalar_type(value))
        else:
            shape = Constant(())
            dtype = Constant(type(value))
        self.set_details(expression, shape=shape, dtype=dtype)

    def visit_Divide(self, expression: expressions.Divide):
        self.visit(expression.expr1)
        self.visit(expression.expr2)
        shape_assertions = [
            Constant(_broadcastable)(
                self.shapes[expression.expr1], self.shapes[expression.expr2]
            )
        ]

        if not _check_assertions(shape_assertions):
            raise DNNVShapeError(
                f"Incompatible shapes in Divide: {self.shapes[expression.expr1].value} and {self.shapes[expression.expr2].value}"
            )
        self.assertions.extend(shape_assertions)

        self.set_details(
            expression,
            shape=Constant(np.broadcast_shapes)(
                self.shapes[expression.expr1], self.shapes[expression.expr2]
            ),
            dtype=Constant(np.result_type)(
                self.types[expression.expr1], self.types[expression.expr2]
            ),
        )

    def visit_Equal(self, expression: expressions.Equal):
        self.visit(expression.expr1)
        self.visit(expression.expr2)

        shape_assertions = [
            Constant(_broadcastable)(
                self.shapes[expression.expr1], self.shapes[expression.expr2]
            )
        ]
        if not _check_assertions(shape_assertions):
            raise DNNVShapeError(
                f"Incompatible shapes in Equal: {self.shapes[expression.expr1].value} and {self.shapes[expression.expr2].value}"
            )
        self.assertions.extend(shape_assertions)

        self.set_details(expression, shape=Constant(()), dtype=Constant(bool))

    def visit_Exists(self, expression: expressions.Exists):
        self.visit(expression.expression)
        self.visit(expression.variable)
        shape = self.shapes[expression.expression]
        dtype = self.types[expression.expression]
        self.set_details(expression, shape=shape, dtype=dtype)

    def visit_ExtSlice(self, expression: expressions.ExtSlice):
        for expr in expression.expressions:
            self.visit(expr)
        self.set_details(expression, shape=Constant(()), dtype=Constant(tuple))

    def visit_Forall(self, expression: expressions.Forall):
        self.visit(expression.expression)
        self.visit(expression.variable)
        shape = self.shapes[expression.expression]
        dtype = self.types[expression.expression]
        self.set_details(expression, shape=shape, dtype=dtype)

    def visit_GreaterThan(self, expression: expressions.GreaterThan):
        self.visit(expression.expr1)
        self.visit(expression.expr2)

        shape_assertions = [
            Constant(_broadcastable)(
                self.shapes[expression.expr1], self.shapes[expression.expr2]
            )
        ]
        if not _check_assertions(shape_assertions):
            raise DNNVShapeError(
                f"Incompatible shapes in GreaterThan: {self.shapes[expression.expr1].value} and {self.shapes[expression.expr2].value}"
            )
        self.assertions.extend(shape_assertions)

        self.set_details(expression, shape=Constant(()), dtype=Constant(bool))

    def visit_GreaterThanOrEqual(self, expression: expressions.GreaterThanOrEqual):
        self.visit(expression.expr1)
        self.visit(expression.expr2)

        shape_assertions = [
            Constant(_broadcastable)(
                self.shapes[expression.expr1], self.shapes[expression.expr2]
            )
        ]
        if not _check_assertions(shape_assertions):
            raise DNNVShapeError(
                f"Incompatible shapes in GreaterThanOrEqual: {self.shapes[expression.expr1].value} and {self.shapes[expression.expr2].value}"
            )
        self.assertions.extend(shape_assertions)

        self.set_details(expression, shape=Constant(()), dtype=Constant(bool))

    def visit_IfThenElse(self, expression: expressions.IfThenElse):
        self.visit(expression.condition)
        self.visit(expression.t_expr)
        self.visit(expression.f_expr)
        shape_assertions = [
            self.shapes[expression.condition] == Constant(()),
            self.shapes[expression.t_expr] == self.shapes[expression.f_expr],
            self.shapes[expression.f_expr] == self.shapes[expression.t_expr],
        ]
        type_assertions = [
            self.types[expression.condition] == Constant(bool),
            expressions.Or(
                Constant(np.can_cast)(
                    self.types[expression.t_expr],
                    self.types[expression.f_expr],
                ),
                Constant(np.can_cast)(
                    self.types[expression.f_expr],
                    self.types[expression.t_expr],
                ),
            ),
        ]

        shape = self.shapes[expression.t_expr]
        dtype = Constant(np.result_type)(
            self.types[expression.t_expr], self.types[expression.f_expr]
        )

        if not _check_assertions(shape_assertions[:1]):
            raise DNNVShapeError(
                f"Incompatible shapes in IfThenElse: {self.shapes[expression.condition].value}"
            )
        if not _check_assertions(shape_assertions[1:]):
            raise DNNVShapeError(
                f"Incompatible shapes in IfThenElse: {self.shapes[expression.t_expr].value} and {self.shapes[expression.f_expr].value}"
            )
        self.assertions.extend(shape_assertions)
        if not _check_assertions(type_assertions):
            raise DNNVTypeError("Incompatible types in IfThenElse")
        self.assertions.extend(type_assertions)

        self.set_details(expression, shape=shape, dtype=dtype)

    def visit_Image(self, expression: expressions.Image):
        if expression.is_concrete:
            value = expression.value
            shape: Expression = Constant(np.shape(value))
            dtype: Expression = Constant(_get_type(value))
        else:
            shape = Constant(np.shape)(expression)
            dtype = Constant(_get_type)(expression)
        self.set_details(expression, shape=shape, dtype=dtype)

    def visit_Implies(self, expression: expressions.Implies):
        self.visit(expression.expr1)
        self.visit(expression.expr2)

        type_assertions = [
            self.types[expression.expr1] == Constant(bool),
            self.types[expression.expr2] == Constant(bool),
        ]
        shape_assertions = [
            self.shapes[expression.expr1] == self.shapes[expression.expr2],
            self.shapes[expression.expr2] == self.shapes[expression.expr1],
        ]

        if not _check_assertions(shape_assertions):
            raise DNNVShapeError(
                f"Incompatible shapes in Implies: {self.shapes[expression.expr1].value} and {self.shapes[expression.expr2].value}"
            )
        self.assertions.extend(shape_assertions)
        if not _check_assertions(type_assertions):
            raise DNNVTypeError(
                f"Incompatible types in Implies: {self.types[expression.expr1].value} and {self.types[expression.expr2].value}"
            )
        self.assertions.extend(type_assertions)

        self.set_details(
            expression,
            shape=self.shapes[expression.expr1],
            dtype=Constant(bool),
        )

    def visit_LessThan(self, expression: expressions.LessThan):
        self.visit(expression.expr1)
        self.visit(expression.expr2)

        shape_assertions = [
            Constant(_broadcastable)(
                self.shapes[expression.expr1], self.shapes[expression.expr2]
            )
        ]
        if not _check_assertions(shape_assertions):
            raise DNNVShapeError(
                f"Incompatible shapes in LessThan: {self.shapes[expression.expr1].value} and {self.shapes[expression.expr2].value}"
            )
        self.assertions.extend(shape_assertions)

        self.set_details(expression, shape=Constant(()), dtype=Constant(bool))

    def visit_LessThanOrEqual(self, expression: expressions.LessThanOrEqual):
        self.visit(expression.expr1)
        self.visit(expression.expr2)

        shape_assertions = [
            Constant(_broadcastable)(
                self.shapes[expression.expr1], self.shapes[expression.expr2]
            )
        ]
        if not _check_assertions(shape_assertions):
            raise DNNVShapeError(
                f"Incompatible shapes in LessThanOrEqual: {self.shapes[expression.expr1].value} and {self.shapes[expression.expr2].value}"
            )
        self.assertions.extend(shape_assertions)

        self.set_details(expression, shape=Constant(()), dtype=Constant(bool))

    def visit_Multiply(self, expression: expressions.Multiply):
        dtype = None
        shape = None
        shape_assertions: List[expressions.LogicalExpression] = []
        for expr in expression:
            self.visit(expr)
            if shape is None:
                shape = self.shapes[expr]
            elif shape.is_concrete and self.shapes[expr].is_concrete:
                shape_ = shape.value
                expr_shape = self.shapes[expr].value
                if not _broadcastable(shape_, expr_shape):
                    raise DNNVShapeError(
                        f"Incompatible shapes in Multiply: {shape_} and {expr_shape}"
                    )
                shape_assertions.append(Constant(_broadcastable(shape_, expr_shape)))
                shape = Constant(np.broadcast_shapes(shape_, expr_shape))
            else:
                shape_assertions.append(
                    Constant(_broadcastable)(shape, self.shapes[expr])
                )
                shape = Constant(np.broadcast_shapes)(shape, self.shapes[expr])
            if dtype is None:
                dtype = self.types[expr]
            elif dtype.is_concrete and self.types[expr].is_concrete:
                dtype_ = dtype.value
                expr_dtype = self.types[expr].value
                dtype = Constant(np.result_type(dtype_, expr_dtype))
            else:
                dtype = Constant(np.result_type)(dtype, self.types[expr])

        if not _check_assertions(shape_assertions):
            raise DNNVShapeError(f"Incompatible shapes in Multiply")
        self.assertions.extend(shape_assertions)
        # TODO : check type assertions

        self.set_details(expression, shape=shape, dtype=dtype)

    def visit_Negation(self, expression: expressions.Negation):
        self.visit(expression.expr)
        shape = self.shapes[expression.expr]
        dtype = self.types[expression.expr]
        self.set_details(expression, shape=shape, dtype=dtype)

    def visit_Network(self, expression: expressions.Network):
        self.set_details(expression, shape=Constant(()), dtype=Constant(OperationGraph))

    def visit_Not(self, expression: expressions.Not):
        self.visit(expression.expr)
        type_assertions = [self.types[expression.expr] == Constant(bool)]
        shape = self.shapes[expression.expr]
        dtype = Constant(bool)

        if not _check_assertions(type_assertions):
            raise DNNVTypeError(
                f"Incompatible type in Not: {self.types[expression.expr].value}"
            )
        self.assertions.extend(type_assertions)
        # TODO : shape assertions

        self.set_details(expression, shape=shape, dtype=dtype)

    def visit_NotEqual(self, expression: expressions.NotEqual):
        self.visit(expression.expr1)
        self.visit(expression.expr2)

        shape_assertions = [
            Constant(_broadcastable)(
                self.shapes[expression.expr1], self.shapes[expression.expr2]
            )
        ]
        if not _check_assertions(shape_assertions):
            raise DNNVShapeError(
                f"Incompatible shapes in NotEqual: {self.shapes[expression.expr1].value} and {self.shapes[expression.expr2].value}"
            )
        self.assertions.extend(shape_assertions)

        self.set_details(expression, shape=Constant(()), dtype=Constant(bool))

    def visit_Or(self, expression: expressions.Or):
        shape = None
        shape_assertions: List[expressions.LogicalExpression] = []
        type_assertions: List[expressions.LogicalExpression] = []
        for expr in expression:
            self.visit(expr)
            if shape is None:
                shape = self.shapes[expr]
            elif shape.is_concrete and self.shapes[expr].is_concrete:
                shape_ = shape.value
                expr_shape = self.shapes[expr].value
                if not _broadcastable(shape_, expr_shape):
                    raise DNNVShapeError(
                        f"Incompatible shapes in Or: {shape_} and {expr_shape}"
                    )
                shape_assertions.append(Constant(_broadcastable(shape_, expr_shape)))
                shape = Constant(np.broadcast_shapes(shape_, expr_shape))
            else:
                shape_assertions.append(
                    Constant(_broadcastable)(shape, self.shapes[expr])
                )
                shape = Constant(np.broadcast_shapes)(shape, self.shapes[expr])
            type_assertions.append(self.types[expr] == Constant(bool))

        if not _check_assertions(shape_assertions):
            raise DNNVShapeError(f"Incompatible shapes in Or")
        self.assertions.extend(shape_assertions)
        if not _check_assertions(type_assertions):
            raise DNNVTypeError(f"Incompatible types in Or")
        self.assertions.extend(type_assertions)

        self.set_details(expression, shape=shape, dtype=Constant(bool))

    def visit_Parameter(self, expression: expressions.Parameter):
        dtype = Constant(expression.type)
        if expression.is_concrete:
            shape: Expression = Constant(np.shape(expression.value))
        else:
            shape = Constant(np.shape)(expression)
        self.set_details(expression, shape=shape, dtype=dtype)

    def visit_Slice(self, expression: expressions.Slice):
        self.visit(expression.start)
        self.visit(expression.stop)
        self.visit(expression.step)
        self.set_details(expression, shape=Constant(()), dtype=Constant(slice))

    def visit_Subscript(self, expression: expressions.Subscript):
        self.visit(expression.expr1)
        self.visit(expression.expr2)

        if (
            self.types[expression.expr1].is_concrete
            and isinstance(self.types[expression.expr1].value, type)
            and issubclass(self.types[expression.expr1].value, OperationGraph)
        ):
            shape: Expression = Constant(())
            dtype: Expression = Constant(OperationGraph)
        else:
            shape = Constant(np.shape)(
                Constant(np.empty)(self.shapes[expression.expr1])[expression.expr2]
            )
            dtype = self.types[expression.expr1]
        self.set_details(expression, shape=shape, dtype=dtype)

    def visit_Subtract(self, expression: expressions.Subtract):
        self.visit(expression.expr1)
        self.visit(expression.expr2)
        shape_assertions = [
            Constant(_broadcastable)(
                self.shapes[expression.expr1], self.shapes[expression.expr2]
            )
        ]

        if not _check_assertions(shape_assertions):
            raise DNNVShapeError(
                f"Incompatible shapes in Subtract: {self.shapes[expression.expr1].value} and {self.shapes[expression.expr2].value}"
            )
        self.assertions.extend(shape_assertions)

        self.set_details(
            expression,
            shape=Constant(np.broadcast_shapes)(
                self.shapes[expression.expr1], self.shapes[expression.expr2]
            ),
            dtype=Constant(np.result_type)(
                self.types[expression.expr1], self.types[expression.expr2]
            ),
        )

    def visit_Symbol(self, expression: expressions.Symbol):
        self.set_details(expression)


__all__ = [
    "DetailsInference",
    "DNNVInferenceError",
    "DNNVShapeError",
    "DNNVTypeError",
]
