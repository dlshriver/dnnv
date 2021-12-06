from dnnv.properties import *
from dnnv.properties.transformers import CanonicalTransformer


def test_Negation_symbol():
    transformer = CanonicalTransformer()

    expr = Negation(Symbol("a"))
    new_expr = transformer.visit(expr)
    assert new_expr is not expr
    assert isinstance(new_expr, And)
    assert len(new_expr.expressions) == 1
    assert isinstance(new_expr.expressions[0], Or)
    assert len(new_expr.expressions[0].expressions) == 1
    assert isinstance(new_expr.expressions[0].expressions[0], Add)
    new_expr_add = new_expr.expressions[0].expressions[0]
    assert len(new_expr_add.expressions) == 1
    assert Multiply(Constant(-1), Symbol("a")) in new_expr_add.expressions


def test_Negation_constant():
    transformer = CanonicalTransformer()
    transformer._top_level = False

    expr = Negation(Constant(302))
    new_expr = transformer.visit(expr)
    assert new_expr is not expr
    assert isinstance(new_expr, Add)
    new_expr_add = new_expr
    assert len(new_expr_add.expressions) == 1
    assert Multiply(Constant(1), Constant(-302)) in new_expr_add.expressions
