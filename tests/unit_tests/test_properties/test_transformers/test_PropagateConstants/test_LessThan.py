import numpy as np
import pytest

from dnnv.properties.expressions import *
from dnnv.properties.transformers import PropagateConstants


def test_LessThan_symbols():
    transformer = PropagateConstants()

    a, b = Symbol("a"), Symbol("b")
    expr = LessThan(a, b)
    new_expr = transformer.visit(expr)
    assert new_expr is not expr
    assert isinstance(new_expr, LessThan)
    assert new_expr.expr1 is Symbol("a")
    assert new_expr.expr2 is Symbol("b")


@pytest.mark.xfail
def test_LessThan_same_symbol():
    transformer = PropagateConstants()

    a, b = Symbol("x"), Symbol("x")
    expr = LessThan(a, b)
    new_expr = transformer.visit(expr)
    assert new_expr is not expr
    assert isinstance(new_expr, Constant)
    assert new_expr.value == False


def test_LessThan_consts():
    transformer = PropagateConstants()

    a, b = Constant(6), Constant(2)
    expr = LessThan(a, b)
    new_expr = transformer.visit(expr)
    assert new_expr is not expr
    assert isinstance(new_expr, Constant)
    assert new_expr.value == False

    a, b = Constant(-6), Constant(-2)
    expr = LessThan(a, b)
    new_expr = transformer.visit(expr)
    assert new_expr is not expr
    assert isinstance(new_expr, Constant)
    assert new_expr.value == True

    a, b = Constant(33), Constant(33)
    expr = LessThan(a, b)
    new_expr = transformer.visit(expr)
    assert new_expr is not expr
    assert isinstance(new_expr, Constant)
    assert new_expr.value == False


def test_LessThan_const_arrays():
    transformer = PropagateConstants()

    a, b = Constant(np.full((1, 5), 6)), Constant(np.full((1, 5), 2))
    expr = LessThan(a, b)
    new_expr = transformer.visit(expr)
    assert new_expr is not expr
    assert isinstance(new_expr, Constant)
    assert np.all(new_expr.value == False)

    a, b = Constant(np.full((1, 5), -6)), Constant(np.full((1, 5), -2))
    expr = LessThan(a, b)
    new_expr = transformer.visit(expr)
    assert new_expr is not expr
    assert isinstance(new_expr, Constant)
    assert np.all(new_expr.value == True)

    a, b = Constant(np.full((1, 5), 33)), Constant(np.full((1, 5), 33))
    expr = LessThan(a, b)
    new_expr = transformer.visit(expr)
    assert new_expr is not expr
    assert isinstance(new_expr, Constant)
    assert np.all(new_expr.value == False)

    arr_a = np.array([1, 2, 3])
    arr_b = np.array([1, 0, 4])
    a, b = Constant(arr_a), Constant(arr_b)
    expr = LessThan(a, b)
    new_expr = transformer.visit(expr)
    assert new_expr is not expr
    assert isinstance(new_expr, Constant)
    assert np.all(new_expr.value == np.array([0, 0, 1], dtype=bool))
