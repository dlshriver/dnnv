from dnnv.properties.expressions import *


def test_invert():
    expr = LessThanOrEqual(Symbol("x"), Symbol("y"))
    not_expr = ~expr
    assert isinstance(not_expr, GreaterThan)
    assert not_expr.expr1 is expr.expr1
    assert not_expr.expr2 is expr.expr2
