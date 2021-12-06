from dnnv.properties.expressions import *


def test_invert():
    expr = Implies(Symbol("x"), Symbol("y"))
    not_expr = ~expr
    assert isinstance(not_expr, And)
    for e in not_expr.expressions:
        if isinstance(e, Not):
            assert e.expr is expr.expr2
        else:
            assert e is expr.expr1
