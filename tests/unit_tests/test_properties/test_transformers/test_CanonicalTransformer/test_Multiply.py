from dnnv.properties import *
from dnnv.properties.transformers import CanonicalTransformer


def test_Multiply_symbols():
    transformer = CanonicalTransformer()

    expr = Multiply(Symbol("a"), Symbol("b"), Symbol("c"))
    new_expr = transformer.visit(expr)
    assert new_expr is not expr
    assert len(new_expr.expressions) == 1
    assert (
        Multiply(Constant(1), Symbol("a"), Symbol("b"), Symbol("c"))
        in new_expr.expressions
    )


def test_Multiply_constants():
    transformer = CanonicalTransformer()

    expr = Multiply(Constant(1), Constant(-3))
    new_expr = transformer.visit(expr)
    assert new_expr is not expr
    assert new_expr is Constant(-3)


def test_Multiply_mixed_constants_symbols():
    transformer = CanonicalTransformer()

    expr = Multiply(Constant(1), Symbol("a"), Constant(-3), Symbol("b"))
    new_expr = transformer.visit(expr)
    assert new_expr is not expr
    assert len(new_expr.expressions) == 1
    assert Multiply(Constant(-3), Symbol("a"), Symbol("b")) in new_expr.expressions


def test_Multiply_mixed_additions():
    transformer = CanonicalTransformer()

    expr = Multiply(
        Add(Constant(1), Symbol("a")),
        Add(Constant(-3), Symbol("b"), Symbol("a")),
    )
    new_expr = transformer.visit(expr)
    assert new_expr is not expr
    assert len(new_expr.expressions) == 5
    assert Multiply(Constant(1), Constant(-3)) in new_expr
    assert Multiply(Constant(-2), Symbol("a")) in new_expr
    assert Multiply(Constant(1), Symbol("b")) in new_expr
    assert Multiply(Constant(1), Symbol("a"), Symbol("b")) in new_expr
    assert Multiply(Constant(1), Symbol("a"), Symbol("a")) in new_expr
