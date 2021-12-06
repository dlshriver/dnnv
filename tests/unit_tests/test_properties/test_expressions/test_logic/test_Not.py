from dnnv.properties.expressions import Not, Symbol


def test_invert():
    expr = Symbol("x")
    not_expr = ~expr
    assert isinstance(not_expr, Not)
    assert not_expr.expr is expr
    not_not_expr = ~not_expr
    assert not_not_expr is expr
