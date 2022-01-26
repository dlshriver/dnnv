from dnnv.properties import *
from dnnv.properties.transformers import CanonicalTransformer


def test_Subtract_symbols():
    transformer = CanonicalTransformer()

    a, b = Symbol("a"), Symbol("b")
    expr = Subtract(a, b)
    new_expr = transformer.visit(expr)
    assert new_expr is not expr
    assert isinstance(new_expr, Add)
    assert Multiply(Constant(1), a) in new_expr.expressions
    assert Multiply(Constant(-1), b) in new_expr.expressions


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
