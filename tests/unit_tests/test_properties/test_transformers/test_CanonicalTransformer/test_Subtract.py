from dnnv.properties import *
from dnnv.properties.transformers import CanonicalTransformer


def test_Subtract_symbols():
    transformer = CanonicalTransformer()

    a, b = Symbol("a"), Symbol("b")
    expr = Subtract(a, b)
    new_expr = transformer.visit(expr)
    assert new_expr is not expr
    assert isinstance(new_expr, And)
    assert len(new_expr.expressions) == 1
    assert isinstance(new_expr.expressions[0], Or)
    assert len(new_expr.expressions[0].expressions) == 1
    assert isinstance(new_expr.expressions[0].expressions[0], Add)
    new_expr_add = new_expr.expressions[0].expressions[0]
    assert len(new_expr_add.expressions) == 2
    assert Multiply(Constant(1), a) in new_expr_add.expressions
    assert Multiply(Constant(-1), b) in new_expr_add.expressions


def test_Subtract_constants():
    transformer = CanonicalTransformer()
    transformer._top_level = False

    expr = Subtract(Constant(402), Constant(200))
    new_expr = transformer.visit(expr)
    assert new_expr is not expr
    assert isinstance(new_expr, Add)
    new_expr_add = new_expr
    assert len(new_expr_add.expressions) == 1
    assert Multiply(Constant(1), Constant(202)) in new_expr_add.expressions
