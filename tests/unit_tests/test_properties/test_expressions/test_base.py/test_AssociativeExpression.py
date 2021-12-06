import pytest

from dnnv.properties.expressions import *


def test_AssociativeExpression():
    expr = AssociativeExpression(Symbol("a"), Symbol("b"), Symbol("c"))
    assert expr.expressions == [Symbol("a"), Symbol("b"), Symbol("c")]

    expr = AssociativeExpression(
        AssociativeExpression(Symbol("x"), Symbol("y")), Symbol("z")
    )
    assert expr.expressions == [Symbol("x"), Symbol("y"), Symbol("z")]


def test_repr():
    expr = AssociativeExpression(Symbol("a"), Symbol("b"), Symbol("c"))
    assert repr(expr) == "AssociativeExpression(Symbol('a'), Symbol('b'), Symbol('c'))"

    expr = AssociativeExpression(
        AssociativeExpression(Symbol("x"), Symbol("y")), Symbol("z")
    )
    assert repr(expr) == "AssociativeExpression(Symbol('x'), Symbol('y'), Symbol('z'))"


def test_str():
    expr = AssociativeExpression(Symbol("a"), Symbol("b"), Symbol("c"))
    expr.OPERATOR_SYMBOL = ":"
    assert str(expr) == "(a : b : c)"

    expr = AssociativeExpression(
        AssociativeExpression(Symbol("x"), Symbol("y")), Symbol("z")
    )
    expr.OPERATOR_SYMBOL = "#"
    assert str(expr) == "(x # y # z)"


def test_iter():
    expr = AssociativeExpression(Symbol("a"), Symbol("b"), Symbol("c"))
    assert list(iter(expr)) == [Symbol("a"), Symbol("b"), Symbol("c")]

    expr = AssociativeExpression(
        AssociativeExpression(Symbol("x"), Symbol("y")), Symbol("z")
    )
    assert list(iter(expr)) == [Symbol("x"), Symbol("y"), Symbol("z")]


def test_is_equivalent():
    expr1 = AssociativeExpression(Symbol("a"), Symbol("b"), Symbol("c"))
    expr2 = AssociativeExpression(Symbol("a"), Symbol("b"), Symbol("c"))
    expr3 = AssociativeExpression(Symbol("c"), Symbol("a"), Symbol("b"))
    expr4 = AssociativeExpression(Symbol("a"), Symbol("a"), Symbol("b"))

    assert expr1.is_equivalent(expr1)
    assert expr1.is_equivalent(expr2)
    assert expr2.is_equivalent(expr1)
    assert expr1.is_equivalent(expr3)
    assert expr3.is_equivalent(expr1)
    assert not expr1.is_equivalent(expr4)
    assert not expr4.is_equivalent(expr1)


def test_Expression_value():
    expr = AssociativeExpression(Symbol("a"), Symbol("b"), Symbol("c"))

    with pytest.raises(ValueError, match="Cannot get value of non-concrete expression"):
        _ = expr.value
