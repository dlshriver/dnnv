import pytest

from dnnv.properties.expressions import *


def test_Subscript():
    expr = Subscript(Symbol("a"), Symbol("b"))
    assert expr.expr1 == Symbol("a")
    assert expr.expr2 == Symbol("b")

    expr = Subscript(Constant((1, 2)), Constant(0))
    assert expr.expr1 == Constant((1, 2))
    assert expr.expr2 == Constant(0)


def test_expr():
    expr = Subscript(Symbol("a"), Symbol("b"))
    assert expr.expr == Symbol("a")

    expr = Subscript(Constant((1, 2)), Constant(0))
    assert expr.expr == Constant((1, 2))


def test_index():
    expr = Subscript(Symbol("a"), Symbol("b"))
    assert expr.index == Symbol("b")

    expr = Subscript(Constant((1, 2)), Constant(0))
    assert expr.index == Constant(0)


def test_value():
    expr = Subscript(Symbol("a"), Symbol("b"))
    with pytest.raises(ValueError):
        _ = expr.value

    expr = Subscript(Constant((1, 2)), Constant(0))
    assert expr.value == 1


def test_repr():
    expr = Subscript(Symbol("a"), Symbol("b"))
    assert repr(expr) == "Symbol('a')[Symbol('b')]"

    expr = Subscript(Constant((1, 2)), Constant(0))
    assert repr(expr) == "(1, 2)[0]"


def test_str():
    expr = Subscript(Symbol("a"), Symbol("b"))
    assert str(expr) == "a[b]"

    expr = Subscript(Constant((1, 2)), Constant(0))
    assert str(expr) == "(1, 2)[0]"
