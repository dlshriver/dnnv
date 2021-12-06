from dnnv.properties import *
from dnnv.properties.transformers import CanonicalTransformer


def test_Add_symbols():
    transformer = CanonicalTransformer()

    expr = Add(Symbol("a"), Symbol("b"), Symbol("c"))
    new_expr = transformer.visit(expr)
    assert new_expr is not expr
    assert isinstance(new_expr, And)
    assert len(new_expr.expressions) == 1
    assert isinstance(new_expr.expressions[0], Or)
    assert len(new_expr.expressions[0].expressions) == 1
    assert isinstance(new_expr.expressions[0].expressions[0], Add)
    new_expr_add = new_expr.expressions[0].expressions[0]
    assert len(new_expr_add.expressions) == 3
    assert Multiply(Constant(1), Symbol("a")) in new_expr_add.expressions
    assert Multiply(Constant(1), Symbol("b")) in new_expr_add.expressions
    assert Multiply(Constant(1), Symbol("c")) in new_expr_add.expressions


def test_Add_constants():
    transformer = CanonicalTransformer()

    expr = Add(Constant(1), Constant(-3))
    new_expr = transformer.visit(expr)
    assert new_expr is not expr
    assert isinstance(new_expr, And)
    assert len(new_expr.expressions) == 1
    assert isinstance(new_expr.expressions[0], Or)
    assert len(new_expr.expressions[0].expressions) == 1
    assert isinstance(new_expr.expressions[0].expressions[0], Constant)
    new_expr_const = new_expr.expressions[0].expressions[0]
    assert new_expr_const is Constant(-2)


def test_Add_mixed_constants_symbols():
    transformer = CanonicalTransformer()

    expr = Add(Constant(1), Symbol("a"), Constant(-3), Symbol("b"))
    new_expr = transformer.visit(expr)
    assert new_expr is not expr
    assert isinstance(new_expr, And)
    assert len(new_expr.expressions) == 1
    assert isinstance(new_expr.expressions[0], Or)
    assert len(new_expr.expressions[0].expressions) == 1
    assert isinstance(new_expr.expressions[0].expressions[0], Add)
    new_expr_add = new_expr.expressions[0].expressions[0]
    assert len(new_expr_add.expressions) == 3
    assert Multiply(Constant(1), Symbol("a")) in new_expr_add.expressions
    assert Multiply(Constant(1), Symbol("b")) in new_expr_add.expressions
    assert Multiply(Constant(1), Constant(-2)) in new_expr_add.expressions


def test_Add_multiplications_constant():
    transformer = CanonicalTransformer()

    expr = Add(
        Multiply(Constant(1), Constant(5), Constant(10)),
        Multiply(Constant(-3), Symbol("b")),
        Multiply(Constant(2), Symbol("a")),
        Symbol("b"),
    )
    new_expr = transformer.visit(expr)
    assert new_expr is not expr
    assert isinstance(new_expr, And)
    assert len(new_expr.expressions) == 1
    assert isinstance(new_expr.expressions[0], Or)
    assert len(new_expr.expressions[0].expressions) == 1
    assert isinstance(new_expr.expressions[0].expressions[0], Add)
    new_expr_add = new_expr.expressions[0].expressions[0]
    assert len(new_expr_add.expressions) == 3
    assert Multiply(Constant(2), Symbol("a")) in new_expr_add.expressions
    assert Multiply(Constant(-2), Symbol("b")) in new_expr_add.expressions
    assert Multiply(Constant(1), Constant(50)) in new_expr_add.expressions


def test_Add_multiplications_linear():
    transformer = CanonicalTransformer()

    expr = Add(
        Multiply(Constant(1), Symbol("a")),
        Multiply(Constant(-3), Symbol("b")),
        Multiply(Constant(2), Symbol("a")),
        Symbol("b"),
    )
    new_expr = transformer.visit(expr)
    assert new_expr is not expr
    assert isinstance(new_expr, And)
    assert len(new_expr.expressions) == 1
    assert isinstance(new_expr.expressions[0], Or)
    assert len(new_expr.expressions[0].expressions) == 1
    assert isinstance(new_expr.expressions[0].expressions[0], Add)
    new_expr_add = new_expr.expressions[0].expressions[0]
    assert len(new_expr_add.expressions) == 2
    assert Multiply(Constant(3), Symbol("a")) in new_expr_add.expressions
    assert Multiply(Constant(-2), Symbol("b")) in new_expr_add.expressions


def test_Add_multiplications_nonlinear():
    transformer = CanonicalTransformer()

    expr = Add(
        Multiply(Constant(1), Symbol("a"), Symbol("b")),
        Multiply(Constant(-3), Symbol("b")),
        Multiply(Constant(2), Symbol("a")),
        Symbol("b"),
    )
    new_expr = transformer.visit(expr)
    assert new_expr is not expr
    assert isinstance(new_expr, And)
    assert len(new_expr.expressions) == 1
    assert isinstance(new_expr.expressions[0], Or)
    assert len(new_expr.expressions[0].expressions) == 1
    assert isinstance(new_expr.expressions[0].expressions[0], Add)
    new_expr_add = new_expr.expressions[0].expressions[0]
    assert len(new_expr_add.expressions) == 3
    assert Multiply(Constant(2), Symbol("a")) in new_expr_add.expressions
    assert Multiply(Constant(-2), Symbol("b")) in new_expr_add.expressions
    assert Multiply(Constant(1), Symbol("a"), Symbol("b")) in new_expr_add.expressions


def test_Add_multiplication_empty():
    transformer = CanonicalTransformer()
    # Empty expressions will already be simplified after propagate_constants
    # TODO : is this test really necessary?
    transformer._top_level = False

    expr = Add(
        Multiply(),
        Multiply(Constant(-3), Symbol("b")),
        Multiply(Constant(2), Symbol("a")),
        Constant(2),
    )
    new_expr = transformer.visit(expr)
    assert new_expr is not expr
    assert isinstance(new_expr, Add)
    new_expr_add = new_expr
    assert len(new_expr_add.expressions) == 3
    assert Multiply(Constant(2), Symbol("a")) in new_expr_add.expressions
    assert Multiply(Constant(-3), Symbol("b")) in new_expr_add.expressions
    assert Multiply(Constant(1), Constant(3)) in new_expr_add.expressions


def test_Add_multiplication_zero():
    transformer = CanonicalTransformer()

    expr = Add(
        Multiply(Constant(-2), Symbol("a")),
        Multiply(Constant(-3), Symbol("b")),
        Multiply(Constant(2), Symbol("a")),
        Constant(2),
    )
    new_expr = transformer.visit(expr)
    assert new_expr is not expr
    assert isinstance(new_expr, And)
    assert len(new_expr.expressions) == 1
    assert isinstance(new_expr.expressions[0], Or)
    assert len(new_expr.expressions[0].expressions) == 1
    assert isinstance(new_expr.expressions[0].expressions[0], Add)
    new_expr_add = new_expr.expressions[0].expressions[0]
    assert len(new_expr_add.expressions) == 2
    assert Multiply(Constant(-3), Symbol("b")) in new_expr_add.expressions
    assert Multiply(Constant(1), Constant(2)) in new_expr_add.expressions


def test_Add_additions():
    transformer = CanonicalTransformer()
    # Adds will already be simplified after propagate_constants
    # TODO : is this test really necessary?
    transformer._top_level = False

    expr = Add(
        Multiply(Constant(-3), Symbol("b")),
        Multiply(Constant(2), Symbol("a")),
        Symbol("b"),
    )
    expr.expressions.append(Add(Constant(1), Symbol("a"), Symbol("b")))
    new_expr = transformer.visit(expr)
    assert isinstance(new_expr, Add)
    new_expr_add = new_expr
    assert len(new_expr_add.expressions) == 3
    assert Multiply(Constant(3), Symbol("a")) in new_expr_add.expressions
    assert Multiply(Constant(-1), Symbol("b")) in new_expr_add.expressions
    assert Multiply(Constant(1), Constant(1)) in new_expr_add.expressions
