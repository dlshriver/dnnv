from dnnv.properties import *
from dnnv.properties.transformers import CanonicalTransformer


def test_LessThan_symbols():
    transformer = CanonicalTransformer()

    expr = LessThan(Symbol("a"), Symbol("b"))
    new_expr = transformer.visit(expr)
    assert new_expr is not expr
    assert isinstance(new_expr, Or)
    assert len(new_expr.expressions) == 1
    assert isinstance(new_expr.expressions[0], And)
    assert len(new_expr.expressions[0].expressions) == 1
    assert isinstance(new_expr.expressions[0].expressions[0], LessThan)
    new_expr_lt = new_expr.expressions[0].expressions[0]
    assert isinstance(new_expr_lt.expr1, Add)
    assert len(new_expr_lt.expr1.expressions) == 2
    assert Multiply(Constant(1), Symbol("a")) in new_expr_lt.expr1.expressions
    assert Multiply(Constant(-1), Symbol("b")) in new_expr_lt.expr1.expressions
    assert isinstance(new_expr_lt.expr2, Constant)
    assert new_expr_lt.expr2 is Constant(0)


def test_LessThan_constants():
    transformer = CanonicalTransformer()

    expr = LessThan(Constant(9), Constant(5))
    new_expr = transformer.visit(expr)
    assert new_expr is not expr
    assert new_expr is Constant(False)


def test_LessThan_mixed_symbols_constants_0():
    transformer = CanonicalTransformer()

    a, b = Symbol("a"), Symbol("b")
    expr = LessThan(a + b, Constant(5))
    new_expr = transformer.visit(expr)
    assert new_expr is not expr
    assert isinstance(new_expr, Or)
    assert len(new_expr.expressions) == 1
    assert isinstance(new_expr.expressions[0], And)
    assert len(new_expr.expressions[0].expressions) == 1
    assert isinstance(new_expr.expressions[0].expressions[0], LessThan)
    new_expr_lt = new_expr.expressions[0].expressions[0]
    assert isinstance(new_expr_lt.expr1, Add)
    assert len(new_expr_lt.expr1.expressions) == 2
    assert Multiply(Constant(1), Symbol("a")) in new_expr_lt.expr1.expressions
    assert Multiply(Constant(1), Symbol("b")) in new_expr_lt.expr1.expressions
    assert isinstance(new_expr_lt.expr2, Constant)
    assert new_expr_lt.expr2 is Constant(5)


def test_LessThan_mixed_symbols_constants_1():
    transformer = CanonicalTransformer()

    a, b = Symbol("a"), Symbol("b")
    expr = LessThan(Constant(5), a + b)
    new_expr = transformer.visit(expr)
    assert new_expr is not expr
    assert isinstance(new_expr, Or)
    assert len(new_expr.expressions) == 1
    assert isinstance(new_expr.expressions[0], And)
    assert len(new_expr.expressions[0].expressions) == 1
    assert isinstance(new_expr.expressions[0].expressions[0], LessThan)
    new_expr_lt = new_expr.expressions[0].expressions[0]
    assert isinstance(new_expr_lt.expr1, Add)
    assert len(new_expr_lt.expr1.expressions) == 2
    assert Multiply(Constant(-1), Symbol("a")) in new_expr_lt.expr1.expressions
    assert Multiply(Constant(-1), Symbol("b")) in new_expr_lt.expr1.expressions
    assert isinstance(new_expr_lt.expr2, Constant)
    assert new_expr_lt.expr2 is Constant(-5)
