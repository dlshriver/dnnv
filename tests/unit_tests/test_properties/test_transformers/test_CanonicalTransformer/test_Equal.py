from dnnv.properties import *
from dnnv.properties.transformers import CanonicalTransformer


def test_Equal_symbols():
    transformer = CanonicalTransformer()

    a, b = Symbol("a"), Symbol("b")
    expr = Equal(a, b)
    new_expr = transformer.visit(expr)
    assert new_expr is not expr
    assert isinstance(new_expr, And)
    assert len(new_expr.expressions) == 2
    assert isinstance(new_expr.expressions[0], Or)
    assert len(new_expr.expressions[0].expressions) == 1
    assert isinstance(new_expr.expressions[1], Or)
    assert len(new_expr.expressions[1].expressions) == 1
    assert (
        Or(
            GreaterThanOrEqual(
                Add(Multiply(Constant(1), a), Multiply(Constant(-1), b)), Constant(0)
            )
        )
        in new_expr.expressions
    )
    assert (
        Or(
            GreaterThanOrEqual(
                Add(Multiply(Constant(-1), a), Multiply(Constant(1), b)), Constant(0)
            )
        )
        in new_expr.expressions
    )


def test_Equal_constants():
    transformer = CanonicalTransformer()
    transformer._top_level = False

    expr = Equal(Constant(302), Constant(120))
    new_expr = transformer.visit(expr)
    assert new_expr is not expr
    assert isinstance(new_expr, And)
    new_expr_add = new_expr
    assert len(new_expr_add.expressions) == 2
    assert isinstance(new_expr.expressions[0], Or)
    assert len(new_expr.expressions[0].expressions) == 1
    assert isinstance(new_expr.expressions[1], Or)
    assert len(new_expr.expressions[1].expressions) == 1
    assert Or(GreaterThanOrEqual(Add(), Constant(-182))) in new_expr.expressions
    assert Or(GreaterThanOrEqual(Add(), Constant(182))) in new_expr.expressions
