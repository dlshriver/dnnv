import numpy as np

from dnnv.properties.expressions import *
from dnnv.properties.transformers import PropagateConstants


def test_Not_on_symbol():
    transformer = PropagateConstants()

    expr = Not(Symbol("x"))
    new_expr = transformer.visit(expr)
    assert new_expr is not expr
    assert isinstance(new_expr, Not)
    assert new_expr.expr is Symbol("x")


def test_Not_on_const():
    transformer = PropagateConstants()

    expr = Not(Not(Constant(True)))
    new_expr = transformer.visit(expr)
    assert new_expr is not expr
    assert isinstance(new_expr, Constant)
    assert new_expr.value == True

    expr = Not(Constant(True))
    new_expr = transformer.visit(expr)
    assert new_expr is not expr
    assert isinstance(new_expr, Constant)
    assert new_expr.value == False


def test_Not_on_const_array():
    transformer = PropagateConstants()

    arr = np.array([1, 0], dtype=bool)
    expr = Not(Constant(arr))
    new_expr = transformer.visit(expr)
    assert new_expr is not expr
    assert isinstance(new_expr, Constant)
    assert np.allclose(new_expr.value, np.array([0, 1], dtype=bool))
