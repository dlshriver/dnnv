from dnnv.properties.base import *


def test_invert():
    expr = GreaterThan(Symbol("x"), Symbol("y"))
    not_expr = ~expr
    assert isinstance(not_expr, LessThanOrEqual)
    assert not_expr.expr1 is expr.expr1
    assert not_expr.expr2 is expr.expr2
