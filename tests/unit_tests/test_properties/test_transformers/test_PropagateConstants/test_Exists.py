from dnnv.properties.expressions import *
from dnnv.properties.transformers import PropagateConstants


def test_Exists_symbols():
    transformer = PropagateConstants()

    expr = Exists(Symbol("x"), Symbol("x"))
    new_expr = transformer.visit(expr)
    assert new_expr is not expr
    assert isinstance(new_expr, Exists)
    assert new_expr.variable is Symbol("x")
    assert new_expr.expression is Symbol("x")


def test_Exists_consts():
    transformer = PropagateConstants()

    expr = Exists(Symbol("x"), Constant(False))
    new_expr = transformer.visit(expr)
    assert new_expr is not expr
    assert isinstance(new_expr, Constant)
    assert new_expr.value == False

    expr = Exists(Symbol("x"), Constant(True))
    new_expr = transformer.visit(expr)
    assert new_expr is not expr
    assert isinstance(new_expr, Constant)
    assert new_expr.value == True
