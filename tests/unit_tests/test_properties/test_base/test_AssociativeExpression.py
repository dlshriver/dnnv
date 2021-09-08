from dnnv.properties.base import *


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
    expr.OPERATOR = ":"
    assert str(expr) == "(a : b : c)"

    expr = AssociativeExpression(
        AssociativeExpression(Symbol("x"), Symbol("y")), Symbol("z")
    )
    expr.OPERATOR = "#"
    assert str(expr) == "(x # y # z)"


def test_iter():
    expr = AssociativeExpression(Symbol("a"), Symbol("b"), Symbol("c"))
    assert list(iter(expr)) == [Symbol("a"), Symbol("b"), Symbol("c")]

    expr = AssociativeExpression(
        AssociativeExpression(Symbol("x"), Symbol("y")), Symbol("z")
    )
    assert list(iter(expr)) == [Symbol("x"), Symbol("y"), Symbol("z")]
