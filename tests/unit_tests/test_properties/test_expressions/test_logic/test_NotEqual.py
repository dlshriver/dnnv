from dnnv.properties.expressions import Constant, Equal, NotEqual, Symbol


def test_invert():
    expr = NotEqual(Symbol("x"), Symbol("y"))
    not_expr = ~expr
    assert isinstance(not_expr, Equal)
    assert not_expr.expr1 is expr.expr1
    assert not_expr.expr2 is expr.expr2


def test_bool():
    expr = NotEqual(Symbol("x"), Symbol("y"))
    assert bool(expr)

    expr = NotEqual(Symbol("x"), Symbol("x"))
    assert not bool(expr)

    expr = NotEqual(Symbol("x")[0], Symbol("x")[0])
    assert not bool(expr)

    expr = NotEqual(Constant("x"), Constant("x"))
    assert not bool(expr)

    expr = NotEqual(Constant("x"), Constant("y"))
    assert bool(expr)
