from dnnv.properties.base import *


def test_neg():
    expr = Negation(Symbol("x"))
    assert expr.expr is Symbol("x")
    assert -expr is Symbol("x")
