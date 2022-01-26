from dnnv.properties import expressions
from dnnv.properties.expressions.terms.constant import Constant
from dnnv.properties.visitors import ExpressionVisitor


def test_ExpressionVisitor():
    visitor = ExpressionVisitor()

    visitor.visit(expressions.Symbol("A"))
    visitor.visit(
        expressions.And(
            Constant(False),
            expressions.Or(
                expressions.Symbol("B"),
                expressions.Implies(expressions.Symbol("C"), expressions.Symbol("D")),
            ),
        )
    )
