import pytest

from dnnv.properties.expressions import *


def test_BinaryExpression():
    expr = BinaryExpression(Symbol("a"), Symbol("b"))
    assert expr.expr1 == Symbol("a")
    assert expr.expr2 == Symbol("b")


def test_repr():
    expr = BinaryExpression(Symbol("a"), Symbol("b"))
    assert repr(expr) == "BinaryExpression(Symbol('a'), Symbol('b'))"


def test_str():
    expr = BinaryExpression(Symbol("a"), Symbol("b"))
    expr.OPERATOR_SYMBOL = "$"
    assert str(expr) == "(a $ b)"


def test_is_equivalent():
    expr1 = BinaryExpression(Symbol("a"), Symbol("b"))
    expr2 = BinaryExpression(Symbol("a"), Symbol("b"))
    expr3 = BinaryExpression(Symbol("b"), Symbol("a"))
    expr4 = BinaryExpression(Symbol("a"), Symbol("a"))

    assert expr1.is_equivalent(expr1)
    assert expr1.is_equivalent(expr2)
    assert expr2.is_equivalent(expr1)
    assert not expr1.is_equivalent(expr3)
    assert not expr3.is_equivalent(expr1)
    assert not expr1.is_equivalent(expr4)
    assert not expr4.is_equivalent(expr1)


def test_Expression_value():
    expr = BinaryExpression(Symbol("a"), Symbol("b"))

    with pytest.raises(ValueError, match="Cannot get value of non-concrete expression"):
        _ = expr.value
