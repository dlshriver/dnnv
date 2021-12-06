import pytest

from dnnv.properties.expressions import *


def test_TernaryExpression():
    expr = TernaryExpression(Symbol("a"), Symbol("b"), Symbol("c"))
    assert expr.expr1 == Symbol("a")
    assert expr.expr2 == Symbol("b")
    assert expr.expr3 == Symbol("c")


def test_repr():
    expr = TernaryExpression(Symbol("a"), Symbol("b"), Symbol("c"))
    assert repr(expr) == "TernaryExpression(Symbol('a'), Symbol('b'), Symbol('c'))"


def test_str():
    expr = TernaryExpression(Symbol("a"), Symbol("b"), Symbol("c"))
    assert str(expr) == "TernaryExpression(a, b, c)"


def test_is_equivalent():
    expr1 = TernaryExpression(Symbol("a"), Symbol("b"), Symbol("c"))
    expr2 = TernaryExpression(Symbol("a"), Symbol("b"), Symbol("c"))
    expr3 = TernaryExpression(Symbol("c"), Symbol("a"), Symbol("b"))
    expr4 = TernaryExpression(Symbol("a"), Symbol("a"), Symbol("b"))

    assert expr1.is_equivalent(expr1)
    assert expr1.is_equivalent(expr2)
    assert expr2.is_equivalent(expr1)
    assert not expr1.is_equivalent(expr3)
    assert not expr3.is_equivalent(expr1)
    assert not expr1.is_equivalent(expr4)
    assert not expr4.is_equivalent(expr1)


def test_Expression_value():
    expr = TernaryExpression(Symbol("a"), Symbol("b"), Symbol("c"))

    with pytest.raises(ValueError, match="Cannot get value of non-concrete expression"):
        _ = expr.value
