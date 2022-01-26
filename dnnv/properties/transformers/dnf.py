from __future__ import annotations

from typing import Optional, Set, Tuple

from .base import GenericExpressionTransformer
from .lift_ifthenelse import LiftIfThenElse
from .remove_ifthenelse import RemoveIfThenElse
from ..expressions import *
from ..visitors.inference import DetailsInference


class DnfTransformer(GenericExpressionTransformer):
    def visit(self, expression: Expression) -> Expression:
        if self._top_level:
            expression = expression.propagate_constants()
            expression = LiftIfThenElse().visit(expression)
            expression = expression.propagate_constants()
            expression = RemoveIfThenElse().visit(expression)
            expression = expression.propagate_constants()
            if not isinstance(expression, ArithmeticExpression) or isinstance(
                expression, Symbol
            ):
                expression = And(Or(expression))
            DetailsInference().visit(expression)
        expr = super().visit(expression)
        return expr

    def visit_And(self, expression: And) -> Or:
        disjunction: Optional[Or] = None
        expressions: Set[Expression] = set()
        for expr in expression.expressions:
            expr = self.visit(expr)
            if disjunction is None and isinstance(expr, Or):
                disjunction = expr
            else:
                expressions.add(expr)
        if disjunction is None:
            if len(expressions) == 0:
                return Constant(False)
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

    def visit_Exists(self, expression: Exists):
        # TODO : should this do other things ?
        expr = self.visit(expression.expression)
        return expr

    def visit_Forall(self, expression: Forall):
        # TODO : should this do other things ?
        expr = self.visit(expression.expression)
        return expr

    def visit_Implies(self, expression: Implies) -> Or:
        return self.visit(Or(~expression.expr1, expression.expr2))

    def visit_Not(self, expression: Not) -> Or:
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
            if expr.expr1 in expr.ctx.shapes and expr.expr2 in expr.ctx.shapes:
                import numpy as np

                output_shape = np.broadcast_shapes(
                    expr.ctx.shapes[expr.expr1], expr.ctx.shapes[expr.expr2]
                )
                if np.all(np.asarray(output_shape) == 1):
                    return self.visit(
                        Or(
                            opposite_comparison[type(expr)](
                                expr.expr1, expr.expr2, ctx=expr.ctx
                            ),
                            ctx=expr.ctx,
                        )
                    )
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
                return self.visit(
                    Or(
                        *(
                            opposite_comparison[type(expr)](
                                expr1[idx], expr2[idx], ctx=expr.ctx
                            )
                            for idx in np.ndindex(output_shape)
                        ),
                        ctx=expr.ctx,
                    )
                )
        elif isinstance(expression.expr, Symbol):
            return Or(And(Not(expression.expr)))

        result = ~expression.expr
        if isinstance(result, Not):
            return ~self.visit(expression.expr)
        return self.visit(result)

    def visit_Or(self, expression: Or) -> Or:
        expressions: Set[Expression] = set()
        for expr in expression.expressions:
            expr = self.visit(expr)
            if isinstance(expr, Or):
                expressions = expressions.union(expr.expressions)
            else:
                expressions.add(And(expr))
        return Or(*expressions)


__all__ = ["DnfTransformer"]
