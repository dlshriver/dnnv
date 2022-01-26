from dnnv.properties import *
from dnnv.properties.transformers import CanonicalTransformer


def test_Negation_symbol():
    transformer = CanonicalTransformer()

    expr = Negation(Symbol("a"))
    new_expr = transformer.visit(expr)
    assert new_expr is not expr
    assert isinstance(new_expr, Add)
    assert Multiply(Constant(-1), Symbol("a")) in new_expr.expressions


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
