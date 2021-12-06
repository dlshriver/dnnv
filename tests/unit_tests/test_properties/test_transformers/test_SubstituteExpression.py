from dnnv.properties import *
from dnnv.properties.transformers import SubstituteExpression


def test_substitute_symbol():
    a = Symbol("a")
    b = Symbol("b")
    c = Symbol("c")

    expr1 = a + b
    expr2 = a + c

    expr3 = SubstituteExpression(b, c).visit(expr1)

    assert expr3.is_equivalent(expr2)
    assert not expr1.is_equivalent(expr2)


def test_substitute_expression():
    a = Symbol("a")
    b = Symbol("b")
    c = Symbol("c")

    expr = (a + b) / c

    x = Symbol("x")

    e1 = a * ((a + b) / c) + b > c
    e2 = x * x
    e3 = ((a + b) / c) * ((a + b) / c)

    e4 = SubstituteExpression(expr, x).visit(e1)
    assert e4.is_equivalent(a * x + b > c)
    assert not e4.is_equivalent(e1)

    e5 = SubstituteExpression(x, expr).visit(e2)
    assert e5.is_equivalent(e3)

    e6 = SubstituteExpression(expr, x).visit(e3)
    assert e6.is_equivalent(e2)
