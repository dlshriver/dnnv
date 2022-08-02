from __future__ import annotations

from typing import Optional, Set, Tuple, Union

from ..expressions import *
from ..visitors.inference import DetailsInference
from .base import GenericExpressionTransformer
from .lift_ifthenelse import LiftIfThenElse
from .remove_ifthenelse import RemoveIfThenElse
from .substitute_calls import SubstituteCalls


class DnfTransformer(GenericExpressionTransformer):
    def __init__(self):
        super().__init__()
        self._infer_expression_details = lambda: None

    def _lazy_inference(self):
        self._infer_expression_details()
        self._infer_expression_details = lambda: None

    def visit(self, expression: Expression) -> Expression:
        if self._top_level:
            expression = expression.propagate_constants()
            expression = SubstituteCalls().visit(expression)
            expression = expression.propagate_constants()
            expression = LiftIfThenElse().visit(expression)
            expression = expression.propagate_constants()
            expression = RemoveIfThenElse().visit(expression)
            expression = expression.propagate_constants()
            if not isinstance(expression, ArithmeticExpression) or isinstance(
                expression, Symbol
            ):
                expression = Or(And(expression))
            self._infer_expression_details = (
                lambda expression=expression: DetailsInference().visit(expression)
            )
        expression = super().visit(expression)
        return expression

    def visit_And(self, expression: And) -> Union[Constant, Or]:
        disjunction: Optional[Or] = None
        expressions: Set[Expression] = set()
        for expr in expression.expressions:
            expr = self.visit(expr)
            if expr.is_concrete:
                if expr.value:
                    continue
                return Constant(False)
            if disjunction is None and isinstance(expr, Or):
                disjunction = expr
            else:
                expressions.add(expr)
        if disjunction is None:
            if len(expressions) == 0:
                return Constant(True)
            return Or(And(*expressions))
        elif len(expressions) == 0:
            return disjunction
        disjuncts: Set[Tuple[Expression, ...]] = set(
            (e,) for e in disjunction.expressions
        )
        for expr in expressions:
            if isinstance(expr, Or):
                new_disjuncts = set()
                for disjunct in disjuncts:
                    for e in expr.expressions:
                        new_disjuncts.add(tuple(set(disjunct + (e,))))
                disjuncts = new_disjuncts
            else:
                assert not isinstance(expr, And)
                new_disjuncts = set()
                for disjunct in disjuncts:
                    new_disjuncts.add(tuple(set(disjunct + (expr,))))
                disjuncts = new_disjuncts
        return Or(*[And(*disjunct) for disjunct in disjuncts])

    def visit_Exists(self, expression: Exists) -> Union[Constant, Or, Symbol]:
        # TODO : should this do other things ?
        expr = self.visit(expression.expression)
        assert isinstance(expr, (Constant, Or, Symbol))
        return expr

    def visit_Forall(self, expression: Forall) -> Union[Constant, Or, Symbol]:
        # TODO : should this do other things ?
        expr = self.visit(expression.expression)
        assert isinstance(expr, (Constant, Or, Symbol))
        return expr

    def visit_Implies(self, expression: Implies) -> Union[Constant, Or]:
        assert isinstance(expression.expr1, LogicalExpression)
        expr = self.visit(Or(~expression.expr1, expression.expr2))
        assert isinstance(expr, (Constant, Or))
        return expr

    def visit_Not(self, expression: Not) -> Union[Constant, Or]:
        expr = expression.expr
        if isinstance(
            expr,
            (
                Equal,
                NotEqual,
                GreaterThan,
                GreaterThanOrEqual,
                LessThan,
                LessThanOrEqual,
            ),
        ):
            opposite_comparison = {
                Equal: NotEqual,
                NotEqual: Equal,
                GreaterThan: LessThanOrEqual,
                GreaterThanOrEqual: LessThan,
                LessThan: GreaterThanOrEqual,
                LessThanOrEqual: GreaterThan,
            }
            self._lazy_inference()
            if expr.expr1 in expr.ctx.shapes and expr.expr2 in expr.ctx.shapes:
                import numpy as np

                output_shape = np.broadcast_shapes(
                    expr.ctx.shapes[expr.expr1], expr.ctx.shapes[expr.expr2]
                )
                if np.all(np.asarray(output_shape) == 1):
                    expr = self.visit(
                        Or(
                            opposite_comparison[type(expr)](
                                expr.expr1, expr.expr2, ctx=expr.ctx
                            ),
                            ctx=expr.ctx,
                        )
                    )
                    assert isinstance(expr, (Constant, Or))
                    return expr
                expr1 = expr.expr1
                if expr.ctx.shapes[expr1] != output_shape:
                    expr1 = Constant(np.broadcast_to, ctx=expr.ctx)(
                        expr1, Constant(output_shape, ctx=expr.ctx)
                    )
                expr2 = expr.expr2
                if expr.ctx.shapes[expr2] != output_shape:
                    expr2 = Constant(np.broadcast_to, ctx=expr.ctx)(
                        expr2, Constant(output_shape, ctx=expr.ctx)
                    )
                expr = self.visit(
                    Or(
                        *(
                            opposite_comparison[type(expr)](
                                expr1[idx], expr2[idx], ctx=expr.ctx
                            )
                            for idx in np.ndindex(*output_shape)
                        ),
                        ctx=expr.ctx,
                    )
                )
                assert isinstance(expr, (Constant, Or))
                return expr
        elif isinstance(expression.expr, Symbol):
            return Or(And(Not(expression.expr)))

        assert isinstance(expression.expr, LogicalExpression)
        result = ~expression.expr
        if isinstance(result, Not):
            # TODO : should this be an error?
            return result
        expr = self.visit(result)
        assert isinstance(expr, (Constant, Or))
        return expr

    def visit_Or(self, expression: Or) -> Union[Constant, Or]:
        expressions: Set[Expression] = set()
        for expr in expression.expressions:
            expr = self.visit(expr)
            if expr.is_concrete:
                if expr.value:
                    return Constant(True)
                continue
            if isinstance(expr, Or):
                expressions = expressions.union(expr.expressions)
            else:
                expressions.add(And(expr))
        return Or(*expressions)


__all__ = ["DnfTransformer"]
