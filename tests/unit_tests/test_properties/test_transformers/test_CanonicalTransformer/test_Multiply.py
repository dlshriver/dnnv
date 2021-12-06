from dnnv.properties import *
from dnnv.properties.transformers import CanonicalTransformer


def test_Multiply_symbols():
    transformer = CanonicalTransformer()

    expr = Multiply(Symbol("a"), Symbol("b"), Symbol("c"))
    new_expr = transformer.visit(expr)
    assert new_expr is not expr
    assert isinstance(new_expr, And)
    assert len(new_expr.expressions) == 1
    assert isinstance(new_expr.expressions[0], Or)
    assert len(new_expr.expressions[0].expressions) == 1
    assert isinstance(new_expr.expressions[0].expressions[0], Add)
    new_expr_add = new_expr.expressions[0].expressions[0]
    assert len(new_expr_add.expressions) == 1
    assert (
        Multiply(Constant(1), Symbol("a"), Symbol("b"), Symbol("c"))
        in new_expr_add.expressions
    )


def test_Multiply_constants():
    transformer = CanonicalTransformer()

    expr = Multiply(Constant(1), Constant(-3))
    new_expr = transformer.visit(expr)
    assert new_expr is not expr
    assert isinstance(new_expr, And)
    assert len(new_expr.expressions) == 1
    assert isinstance(new_expr.expressions[0], Or)
    assert len(new_expr.expressions[0].expressions) == 1
    assert isinstance(new_expr.expressions[0].expressions[0], Constant)
    new_expr_const = new_expr.expressions[0].expressions[0]
    assert new_expr_const is Constant(-3)


def test_Multiply_mixed_constants_symbols():
    transformer = CanonicalTransformer()

    expr = Multiply(Constant(1), Symbol("a"), Constant(-3), Symbol("b"))
    new_expr = transformer.visit(expr)
    assert new_expr is not expr
    assert isinstance(new_expr, And)
    assert len(new_expr.expressions) == 1
    assert isinstance(new_expr.expressions[0], Or)
    assert len(new_expr.expressions[0].expressions) == 1
    assert isinstance(new_expr.expressions[0].expressions[0], Add)
    new_expr_add = new_expr.expressions[0].expressions[0]
    assert len(new_expr_add.expressions) == 1
    assert Multiply(Constant(-3), Symbol("a"), Symbol("b")) in new_expr_add.expressions


def test_Multiply_mixed_additions():
    transformer = CanonicalTransformer()

    expr = Multiply(
        Add(Constant(1), Symbol("a")),
        Add(Constant(-3), Symbol("b"), Symbol("a")),
    )
    new_expr = transformer.visit(expr)
    assert new_expr is not expr
    assert isinstance(new_expr, And)
    assert len(new_expr.expressions) == 1
    assert isinstance(new_expr.expressions[0], Or)
    assert len(new_expr.expressions[0].expressions) == 1
    assert isinstance(new_expr.expressions[0].expressions[0], Add)
    new_expr_add = new_expr.expressions[0].expressions[0]
    assert len(new_expr_add.expressions) == 5
    assert Multiply(Constant(1), Constant(-3)) in new_expr_add
    assert Multiply(Constant(-2), Symbol("a")) in new_expr_add
    assert Multiply(Constant(1), Symbol("b")) in new_expr_add
    assert Multiply(Constant(1), Symbol("a"), Symbol("b")) in new_expr_add
    assert Multiply(Constant(1), Symbol("a"), Symbol("a")) in new_expr_add
