from dnnv.properties.base import *


def test_BinaryExpression():
    expr = BinaryExpression(Symbol("a"), Symbol("b"))
    assert expr.expr1 == Symbol("a")
    assert expr.expr2 == Symbol("b")


def test_repr():
    expr = BinaryExpression(Symbol("a"), Symbol("b"))
    assert repr(expr) == "BinaryExpression(Symbol('a'), Symbol('b'))"


def test_str():
    expr = BinaryExpression(Symbol("a"), Symbol("b"))
    expr.OPERATOR = "$"
    assert str(expr) == "(a $ b)"
