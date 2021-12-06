import pytest

from dnnv.properties.expressions import *


def test_UnaryExpression():
    expr = UnaryExpression(Symbol("a"))
    assert expr.expr == Symbol("a")


def test_repr():
    expr = UnaryExpression(Symbol("a"))
    assert repr(expr) == "UnaryExpression(Symbol('a'))"


def test_str():
    expr = UnaryExpression(Symbol("a"))
    expr.OPERATOR_SYMBOL = "?"
    assert str(expr) == "?a"


def test_is_equivalent():
    expr1 = UnaryExpression(Symbol("a"))
    expr2 = UnaryExpression(Symbol("a"))
    expr3 = UnaryExpression(Symbol("b"))

    assert expr1.is_equivalent(expr1)
    assert expr1.is_equivalent(expr2)
    assert expr2.is_equivalent(expr1)
    assert not expr1.is_equivalent(expr3)
    assert not expr3.is_equivalent(expr1)


def test_Expression_value():
    expr = UnaryExpression(Symbol("a"))

    with pytest.raises(ValueError, match="Cannot get value of non-concrete expression"):
        _ = expr.value
