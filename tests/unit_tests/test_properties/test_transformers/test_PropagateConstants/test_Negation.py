from dnnv.properties.expressions import *
from dnnv.properties.transformers import PropagateConstants


def test_Negation_on_symbol():
    transformer = PropagateConstants()

    expr = Negation(Symbol("x"))
    new_expr = transformer.visit(expr)
    assert new_expr is not expr
    assert isinstance(new_expr, Negation)
    assert new_expr.expr is Symbol("x")


def test_Negation_on_consts():
    transformer = PropagateConstants()

    expr = Negation(Negation(Constant(1)))
    new_expr = transformer.visit(expr)
    assert new_expr is not expr
    assert isinstance(new_expr, Constant)
    assert new_expr.value == 1

    expr = Negation(Constant(1))
    new_expr = transformer.visit(expr)
    assert new_expr is not expr
    assert isinstance(new_expr, Constant)
    assert new_expr.value == -1
