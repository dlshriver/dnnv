from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ...errors import DNNVError
from ...nn import OperationGraph
from .. import Constant, Context, Expression, expressions, get_context
from ..errors import NonConcreteExpressionError
from .base import ExpressionVisitor


class DNNVInferenceError(DNNVError):
    pass


class DNNVShapeError(DNNVInferenceError):
    pass


class DNNVTypeError(DNNVInferenceError):
    pass


def _check_assertions(assertions: Sequence[Expression]) -> bool:
    for assertion in assertions:
        try:
            if not assertion.value:
                return False
        except NonConcreteExpressionError:
            return True
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
    sub_assertions = []
    for assertion in assertions:
        if isinstance(assertion, expressions.Equal):
            if assertion.is_concrete:
                continue
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
            else:
                sub_assertions.append(assertion)
    if concretized:
        resolve_Equal(sub_assertions)


class DetailsInference(ExpressionVisitor):
    def __init__(self):
        super().__init__()
        self.shapes: Dict[Expression, Expression] = {}
        self.types: Dict[Expression, Expression] = {}
        self.assertions: List[expressions.LogicalExpression] = []
        self.base_ctx: Optional[Context] = None
        self._symbol_count = 0

    def generic_visit(
        self, expression: expressions.Expression
    ) -> expressions.Expression:
        raise NotImplementedError(
            f"DetailsInference is not yet implemented for expression type: {type(expression).__name__}"
        )

    def visit(self, expression) -> Tuple[Expression, Expression]:
        if not isinstance(expression, Expression):
            expression = Constant(expression, ctx=self.base_ctx or get_context())
        if self._top_level:
            with Context():
                self.base_ctx = ctx = expression.ctx
                for expr, shape in ctx.shapes.items():
                    self.shapes[expr] = Constant(shape)
                for expr, dtype in ctx.types.items():
                    self.types[expr] = Constant(dtype)
                result, dtype = super().visit(expression)
                resolve_Equal(self.assertions)
                for k in self.shapes:
                    try:
                        shape = self.shapes[k].value
                        assert isinstance(shape, tuple)
                        ctx.shapes[k] = shape
                    except NonConcreteExpressionError:
                        pass
                    try:
                        dtype = self.types[k].value
                        ctx.types[k] = dtype
                    except NonConcreteExpressionError:
                        pass
            return result, dtype
        assert expression.ctx is self.base_ctx
        result, dtype = super().visit(expression)
        return result, dtype

    def get_symbolic_detail(self, detail_type: str) -> expressions.Symbol:
        self._symbol_count += 1
        return expressions.Symbol(f"!{detail_type}!{self._symbol_count}")

    def set_details(
        self,
        expression,
        shape: Optional[Expression] = None,
        dtype: Optional[Expression] = None,
    ) -> Tuple[Expression, Expression]:
        current_shape = self.shapes.get(expression)
        if current_shape is None:
            if shape is None:
                shape = self.get_symbolic_detail("shape")
            self.shapes[expression] = shape
        elif shape is not None:
            if (
                isinstance(current_shape, expressions.Symbol)
                and not current_shape.is_concrete
            ):
                current_shape.concretize(shape)
            elif shape.value != current_shape.value:
                raise DNNVShapeError(
                    f"Multiple shapes for expression {expression!r}: {shape} and {current_shape}"
                )
        elif shape is None:
            shape = current_shape
        current_dtype = self.types.get(expression)
        if current_dtype is None:
            if dtype is None:
                dtype = self.get_symbolic_detail("dtype")
            self.types[expression] = dtype
        elif dtype is not None:
            if (
                isinstance(current_dtype, expressions.Symbol)
                and not current_dtype.is_concrete
            ):
                current_dtype.concretize(dtype)
            elif dtype.value != current_dtype.value:
                raise DNNVTypeError(
                    f"Multiple types for expression {expression!r}: {dtype} and {current_dtype}"
                )
        elif dtype is None:
            dtype = current_dtype
        assert shape is not None
        assert dtype is not None
        return shape, dtype

    def visit_Add(self, expression: expressions.Add):
        dtype = None
        shape = None
        shape_assertions: List[expressions.LogicalExpression] = []
        for expr in expression:
            expr_shape, expr_dtype = self.visit(expr)
            if shape is None:
                shape = expr_shape
            elif shape.is_concrete and expr_shape.is_concrete:
                shape_ = shape.value
                expr_shape_value = expr_shape.value
                if not _broadcastable(shape_, expr_shape_value):
                    raise DNNVShapeError(
                        f"Incompatible shapes in Add: {shape_} and {expr_shape_value}"
                    )
                shape_assertions.append(
                    Constant(_broadcastable(shape_, expr_shape_value))
                )
                shape = Constant(np.broadcast_shapes(shape_, expr_shape_value))
            else:
                shape_assertions.append(Constant(_broadcastable)(shape, expr_shape))
                shape = Constant(np.broadcast_shapes)(shape, expr_shape)
            if dtype is None:
                dtype = expr_dtype
            elif dtype.is_concrete and expr_dtype.is_concrete:
                dtype_ = dtype.value
                expr_dtype_value = expr_dtype.value
                dtype = Constant(np.result_type(dtype_, expr_dtype_value))
            else:
                dtype = Constant(np.result_type)(dtype, expr_dtype)

        if not _check_assertions(shape_assertions):
            raise DNNVShapeError(f"Incompatible shapes in Add")
        self.assertions.extend(shape_assertions)
        # TODO : check type assertions

        return self.set_details(expression, shape=shape, dtype=dtype)

    def visit_And(self, expression: expressions.And):
        shape = None
        shape_assertions: List[expressions.LogicalExpression] = []
        type_assertions: List[expressions.LogicalExpression] = []
        for expr in expression:
            expr_shape, expr_dtype = self.visit(expr)
            if shape is None:
                shape = expr_shape
            elif shape.is_concrete and expr_shape.is_concrete:
                shape_ = shape.value
                expr_shape_value = expr_shape.value
                if not _broadcastable(shape_, expr_shape_value):
                    raise DNNVShapeError(
                        f"Incompatible shapes in And: {shape_} and {expr_shape_value}"
                    )
                shape_assertions.append(
                    Constant(_broadcastable(shape_, expr_shape_value))
                )
                shape = Constant(np.broadcast_shapes(shape_, expr_shape_value))
            else:
                shape_assertions.append(Constant(_broadcastable)(shape, expr_shape))
                shape = Constant(np.broadcast_shapes)(shape, expr_shape)
            type_assertions.append(expr_dtype == Constant(bool))

        if not _check_assertions(shape_assertions):
            raise DNNVShapeError(f"Incompatible shapes in And")
        self.assertions.extend(shape_assertions)
        if not _check_assertions(type_assertions):
            raise DNNVTypeError(f"Incompatible types in And")
        self.assertions.extend(type_assertions)

        return self.set_details(expression, shape=shape, dtype=Constant(bool))

    def visit_Attribute(self, expression: expressions.Attribute):
        self.visit(expression.expr1)
        self.visit(expression.expr2)

        shape = Constant(np.shape)(expression)
        dtype = Constant(_get_type)(expression)

        return self.set_details(expression, shape=shape, dtype=dtype)

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

        return self.set_details(expression, shape=shape, dtype=dtype)

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
        return self.set_details(expression, shape=shape, dtype=dtype)

    def visit_Divide(self, expression: expressions.Divide):
        expr1_shape, expr1_dtype = self.visit(expression.expr1)
        expr2_shape, expr2_dtype = self.visit(expression.expr2)
        shape_assertions = [Constant(_broadcastable)(expr1_shape, expr2_shape)]

        if not _check_assertions(shape_assertions):
            raise DNNVShapeError(
                f"Incompatible shapes in Divide: {expr1_shape.value} and {expr2_shape.value}"
            )
        self.assertions.extend(shape_assertions)

        return self.set_details(
            expression,
            shape=Constant(np.broadcast_shapes)(expr1_shape, expr2_shape),
            dtype=Constant(np.result_type)(expr1_dtype, expr2_dtype),
        )

    def visit_Equal(self, expression: expressions.Equal):
        expr1_shape, _ = self.visit(expression.expr1)
        expr2_shape, _ = self.visit(expression.expr2)

        shape_assertions = [Constant(_broadcastable)(expr1_shape, expr2_shape)]
        if not _check_assertions(shape_assertions):
            raise DNNVShapeError(
                f"Incompatible shapes in Equal: {expr1_shape.value} and {expr2_shape.value}"
            )
        self.assertions.extend(shape_assertions)

        return self.set_details(expression, shape=Constant(()), dtype=Constant(bool))

    def visit_Exists(self, expression: expressions.Exists):
        shape, dtype = self.visit(expression.expression)
        self.visit(expression.variable)
        return self.set_details(expression, shape=shape, dtype=dtype)

    def visit_ExtSlice(self, expression: expressions.ExtSlice):
        for expr in expression.expressions:
            self.visit(expr)
        return self.set_details(expression, shape=Constant(()), dtype=Constant(tuple))

    def visit_Forall(self, expression: expressions.Forall):
        shape, dtype = self.visit(expression.expression)
        self.visit(expression.variable)
        return self.set_details(expression, shape=shape, dtype=dtype)

    def visit_GreaterThan(self, expression: expressions.GreaterThan):
        expr1_shape, _ = self.visit(expression.expr1)
        expr2_shape, _ = self.visit(expression.expr2)

        shape_assertions = [Constant(_broadcastable)(expr1_shape, expr2_shape)]
        if not _check_assertions(shape_assertions):
            raise DNNVShapeError(
                f"Incompatible shapes in GreaterThan: {expr1_shape.value} and {expr2_shape.value}"
            )
        self.assertions.extend(shape_assertions)

        return self.set_details(expression, shape=Constant(()), dtype=Constant(bool))

    def visit_GreaterThanOrEqual(self, expression: expressions.GreaterThanOrEqual):
        expr1_shape, _ = self.visit(expression.expr1)
        expr2_shape, _ = self.visit(expression.expr2)

        shape_assertions = [Constant(_broadcastable)(expr1_shape, expr2_shape)]
        if not _check_assertions(shape_assertions):
            raise DNNVShapeError(
                f"Incompatible shapes in GreaterThanOrEqual: {expr1_shape.value} and {expr2_shape.value}"
            )
        self.assertions.extend(shape_assertions)

        return self.set_details(expression, shape=Constant(()), dtype=Constant(bool))

    def visit_IfThenElse(self, expression: expressions.IfThenElse):
        cond_shape, cond_dtype = self.visit(expression.condition)
        texpr_shape, texpr_dtype = self.visit(expression.t_expr)
        fexpr_shape, fexpr_dtype = self.visit(expression.f_expr)
        shape_assertions = [
            cond_shape == Constant(()),
            texpr_shape == fexpr_shape,
            fexpr_shape == texpr_shape,
        ]
        type_assertions = [
            cond_dtype == Constant(bool),
            expressions.Or(
                Constant(np.can_cast)(texpr_dtype, fexpr_dtype),
                Constant(np.can_cast)(fexpr_dtype, texpr_dtype),
            ),
        ]

        shape = texpr_shape
        dtype = Constant(np.result_type)(texpr_dtype, fexpr_dtype)

        if not _check_assertions(shape_assertions[:1]):
            raise DNNVShapeError(
                f"Incompatible shapes in IfThenElse: {cond_shape.value}"
            )
        if not _check_assertions(shape_assertions[1:]):
            raise DNNVShapeError(
                f"Incompatible shapes in IfThenElse: {texpr_shape.value} and {fexpr_shape.value}"
            )
        self.assertions.extend(shape_assertions)
        if not _check_assertions(type_assertions):
            raise DNNVTypeError("Incompatible types in IfThenElse")
        self.assertions.extend(type_assertions)

        return self.set_details(expression, shape=shape, dtype=dtype)

    def visit_Image(self, expression: expressions.Image):
        if expression.is_concrete:
            value = expression.value
            shape: Expression = Constant(np.shape(value))
            dtype: Expression = Constant(_get_type(value))
        else:
            shape = Constant(np.shape)(expression)
            dtype = Constant(_get_type)(expression)
        return self.set_details(expression, shape=shape, dtype=dtype)

    def visit_Implies(self, expression: expressions.Implies):
        expr1_shape, expr1_dtype = self.visit(expression.expr1)
        expr2_shape, expr2_dtype = self.visit(expression.expr2)

        type_assertions = [
            expr1_dtype == Constant(bool),
            expr2_dtype == Constant(bool),
        ]
        shape_assertions = [
            expr1_shape == expr2_shape,
            expr2_shape == expr1_shape,
        ]

        if not _check_assertions(shape_assertions):
            raise DNNVShapeError(
                f"Incompatible shapes in Implies: {expr1_shape.value} and {expr2_shape.value}"
            )
        self.assertions.extend(shape_assertions)
        if not _check_assertions(type_assertions):
            raise DNNVTypeError(
                f"Incompatible types in Implies: {expr1_dtype.value} and {expr2_dtype.value}"
            )
        self.assertions.extend(type_assertions)

        return self.set_details(
            expression,
            shape=expr1_shape,
            dtype=Constant(bool),
        )

    def visit_LessThan(self, expression: expressions.LessThan):
        expr1_shape, _ = self.visit(expression.expr1)
        expr2_shape, _ = self.visit(expression.expr2)

        shape_assertions = [Constant(_broadcastable)(expr1_shape, expr2_shape)]
        if not _check_assertions(shape_assertions):
            raise DNNVShapeError(
                f"Incompatible shapes in LessThan: {expr1_shape.value} and {expr2_shape.value}"
            )
        self.assertions.extend(shape_assertions)

        return self.set_details(expression, shape=Constant(()), dtype=Constant(bool))

    def visit_LessThanOrEqual(self, expression: expressions.LessThanOrEqual):
        expr1_shape, _ = self.visit(expression.expr1)
        expr2_shape, _ = self.visit(expression.expr2)

        shape_assertions = [Constant(_broadcastable)(expr1_shape, expr2_shape)]
        if not _check_assertions(shape_assertions):
            raise DNNVShapeError(
                f"Incompatible shapes in LessThanOrEqual: {expr1_shape.value} and {expr2_shape.value}"
            )
        self.assertions.extend(shape_assertions)

        return self.set_details(expression, shape=Constant(()), dtype=Constant(bool))

    def visit_Multiply(self, expression: expressions.Multiply):
        dtype = None
        shape = None
        concrete_shape = None
        shape_assertions: List[expressions.LogicalExpression] = []
        for expr in expression:
            expr_shape, expr_dtype = self.visit(expr)
            if shape is None:
                shape = expr_shape
            elif (
                concrete_shape is not None or shape.is_concrete
            ) and expr_shape.is_concrete:
                if concrete_shape is None:
                    concrete_shape = shape.value
                expr_shape_value = expr_shape.value
                if not _broadcastable(concrete_shape, expr_shape_value):
                    raise DNNVShapeError(
                        f"Incompatible shapes in Multiply: {concrete_shape} and {expr_shape_value}"
                    )
                shape_assertions.append(
                    Constant(_broadcastable(concrete_shape, expr_shape_value))
                )
                concrete_shape = np.broadcast_shapes(concrete_shape, expr_shape_value)
                shape = Constant(concrete_shape)
            else:
                shape_assertions.append(Constant(_broadcastable)(shape, expr_shape))
                shape = Constant(np.broadcast_shapes)(shape, expr_shape)
            if dtype is None:
                dtype = expr_dtype
            elif dtype.is_concrete and expr_dtype.is_concrete:
                dtype_ = dtype.value
                expr_dtype_value = expr_dtype.value
                dtype = Constant(np.result_type(dtype_, expr_dtype_value))
            else:
                dtype = Constant(np.result_type)(dtype, expr_dtype)

        if not _check_assertions(shape_assertions):
            raise DNNVShapeError(f"Incompatible shapes in Multiply")
        self.assertions.extend(shape_assertions)
        # TODO : check type assertions

        return self.set_details(expression, shape=shape, dtype=dtype)

    def visit_Negation(self, expression: expressions.Negation):
        shape, dtype = self.visit(expression.expr)
        return self.set_details(expression, shape=shape, dtype=dtype)

    def visit_Network(self, expression: expressions.Network):
        return self.set_details(
            expression, shape=Constant(()), dtype=Constant(OperationGraph)
        )

    def visit_Not(self, expression: expressions.Not):
        expr_shape, expr_dtype = self.visit(expression.expr)
        type_assertions = [expr_dtype == Constant(bool)]

        if not _check_assertions(type_assertions):
            raise DNNVTypeError(f"Incompatible type in Not: {expr_dtype.value}")
        self.assertions.extend(type_assertions)
        # TODO : shape assertions

        return self.set_details(expression, shape=expr_shape, dtype=Constant(bool))

    def visit_NotEqual(self, expression: expressions.NotEqual):
        expr1_shape, _ = self.visit(expression.expr1)
        expr2_shape, _ = self.visit(expression.expr2)

        shape_assertions = [Constant(_broadcastable)(expr1_shape, expr2_shape)]
        if not _check_assertions(shape_assertions):
            raise DNNVShapeError(
                f"Incompatible shapes in NotEqual: {expr1_shape.value} and {expr2_shape.value}"
            )
        self.assertions.extend(shape_assertions)

        return self.set_details(expression, shape=Constant(()), dtype=Constant(bool))

    def visit_Or(self, expression: expressions.Or):
        shape = None
        shape_assertions: List[expressions.LogicalExpression] = []
        type_assertions: List[expressions.LogicalExpression] = []
        for expr in expression:
            expr_shape, expr_dtype = self.visit(expr)
            if shape is None:
                shape = expr_shape
            elif shape.is_concrete and expr_shape.is_concrete:
                shape_ = shape.value
                expr_shape_value = expr_shape.value
                if not _broadcastable(shape_, expr_shape_value):
                    raise DNNVShapeError(
                        f"Incompatible shapes in Or: {shape_} and {expr_shape_value}"
                    )
                shape_assertions.append(
                    Constant(_broadcastable(shape_, expr_shape_value))
                )
                shape = Constant(np.broadcast_shapes(shape_, expr_shape_value))
            else:
                shape_assertions.append(Constant(_broadcastable)(shape, expr_shape))
                shape = Constant(np.broadcast_shapes)(shape, expr_shape)
            type_assertions.append(expr_dtype == Constant(bool))

        if not _check_assertions(shape_assertions):
            raise DNNVShapeError(f"Incompatible shapes in Or")
        self.assertions.extend(shape_assertions)
        if not _check_assertions(type_assertions):
            raise DNNVTypeError(f"Incompatible types in Or")
        self.assertions.extend(type_assertions)

        return self.set_details(expression, shape=shape, dtype=Constant(bool))

    def visit_Parameter(self, expression: expressions.Parameter):
        dtype = Constant(expression.type)
        if expression.is_concrete:
            shape: Expression = Constant(np.shape(expression.value))
        else:
            shape = Constant(np.shape)(expression)
        return self.set_details(expression, shape=shape, dtype=dtype)

    def visit_Slice(self, expression: expressions.Slice):
        self.visit(expression.start)
        self.visit(expression.stop)
        self.visit(expression.step)
        return self.set_details(expression, shape=Constant(()), dtype=Constant(slice))

    def visit_Subscript(self, expression: expressions.Subscript):
        expr1_shape, expr1_dtype = self.visit(expression.expr1)
        self.visit(expression.expr2)

        if isinstance(expression.expr1, expressions.Network):
            shape: Expression = Constant(())
            dtype: Expression = Constant(OperationGraph)
        else:
            shape = Constant(np.shape)(
                Constant(np.empty)(expr1_shape)[expression.expr2]
            )
            dtype = expr1_dtype
        return self.set_details(expression, shape=shape, dtype=dtype)

    def visit_Subtract(self, expression: expressions.Subtract):
        expr1_shape, expr1_dtype = self.visit(expression.expr1)
        expr2_shape, expr2_dtype = self.visit(expression.expr2)
        shape_assertions = [Constant(_broadcastable)(expr1_shape, expr2_shape)]

        if not _check_assertions(shape_assertions):
            raise DNNVShapeError(
                f"Incompatible shapes in Subtract: {expr1_shape.value} and {expr2_shape.value}"
            )
        self.assertions.extend(shape_assertions)

        return self.set_details(
            expression,
            shape=Constant(np.broadcast_shapes)(expr1_shape, expr2_shape),
            dtype=Constant(np.result_type)(expr1_dtype, expr2_dtype),
        )

    def visit_Symbol(self, expression: expressions.Symbol):
        return self.set_details(expression)


__all__ = [
    "DetailsInference",
    "DNNVInferenceError",
    "DNNVShapeError",
    "DNNVTypeError",
]
