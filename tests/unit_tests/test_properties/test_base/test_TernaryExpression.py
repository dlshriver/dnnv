from dnnv.properties.base import *


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
