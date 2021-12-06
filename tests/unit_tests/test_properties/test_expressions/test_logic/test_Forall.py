from dnnv.properties.expressions import *


def test_invert():
    variable = Symbol("x")
    expression = variable >= Constant(0)
    expr = Forall(variable, expression)
    not_expr = ~expr
    assert isinstance(not_expr, Exists)
    assert not_expr.variable is variable
    assert not_expr.expression == (variable < Constant(0))
