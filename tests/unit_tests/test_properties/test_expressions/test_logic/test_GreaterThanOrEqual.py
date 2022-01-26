from dnnv.properties.expressions import *


def test_invert():
    expr = GreaterThanOrEqual(Symbol("x"), Symbol("y"))
    not_expr = ~expr
    assert isinstance(not_expr, Not)
    assert not_expr.expr is expr
