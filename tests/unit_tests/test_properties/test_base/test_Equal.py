from dnnv.properties.base import Equal, NotEqual, Symbol


def test_invert():
    expr = Equal(Symbol("x"), Symbol("y"))
    not_expr = ~expr
    assert isinstance(not_expr, NotEqual)
    assert not_expr.expr1 is expr.expr1
    assert not_expr.expr2 is expr.expr2


def test_bool():
    expr = Equal(Symbol("x"), Symbol("y"))
    assert not bool(expr)

    expr = Equal(Symbol("x"), Symbol("x"))
    assert bool(expr)

    expr = Equal(Symbol("x")[0], Symbol("x")[0])
    assert bool(expr)
