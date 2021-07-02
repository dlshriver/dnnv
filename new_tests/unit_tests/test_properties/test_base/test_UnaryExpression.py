from dnnv.properties.base import *


def test_UnaryExpression():
    expr = UnaryExpression(Symbol("a"))
    assert expr.expr == Symbol("a")


def test_repr():
    expr = UnaryExpression(Symbol("a"))
    assert repr(expr) == "UnaryExpression(Symbol('a'))"


def test_str():
    expr = UnaryExpression(Symbol("a"))
    expr.OPERATOR = "?"
    assert str(expr) == "?a"
