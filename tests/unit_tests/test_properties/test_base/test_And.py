from dnnv.properties.base import *


def test_invert():
    expr = And(Symbol("x"), Symbol("y"))
    not_expr = ~expr
    assert isinstance(not_expr, Or)
    for e in not_expr.expressions:
        assert isinstance(e, Not)
        assert e.expr in expr.expressions
