import numpy as np
import pytest

from dnnv.properties.expressions import *
from dnnv.properties.transformers import PropagateConstants


def test_Divide_symbols():
    transformer = PropagateConstants()

    a, b = Symbol("a"), Symbol("b")
    expr = a / b
    new_expr = transformer.visit(expr)
    assert new_expr is not expr
    assert isinstance(new_expr, Divide)
    assert new_expr.expr1 is Symbol("a")
    assert new_expr.expr2 is Symbol("b")


def test_Divide_consts():
    transformer = PropagateConstants()

    a, b = Constant(6), Constant(2)
    expr = a / b
    new_expr = transformer.visit(expr)
    assert new_expr is not expr
    assert isinstance(new_expr, Constant)
    assert new_expr.value == 3


def test_Divide_const_arrays():
    transformer = PropagateConstants()

    arr = np.full((1, 3), 4.0)
    a, b = Constant(arr), Constant(2)
    expr = a / b
    new_expr = transformer.visit(expr)
    assert new_expr is not expr
    assert isinstance(new_expr, Constant)
    assert np.allclose(new_expr.value, arr / 2)
    assert np.allclose(a.value, arr)

    arr = np.full((1, 3), 3.0)
    a, b = Constant(arr), Constant(np.ones((1, 3)))
    expr = a / b
    new_expr = transformer.visit(expr)
    assert new_expr is not expr
    assert isinstance(new_expr, Constant)
    assert np.allclose(new_expr.value, arr)
    assert np.allclose(a.value, arr)
    assert np.allclose(b.value, np.ones((1, 3)))

    arr = np.full((1, 3), 3.0)
    a, b = Constant(3), Constant(np.ones((1, 3)))
    expr = a / b
    new_expr = transformer.visit(expr)
    assert new_expr is not expr
    assert isinstance(new_expr, Constant)
    assert np.allclose(new_expr.value, arr)
    assert np.allclose(b.value, np.ones((1, 3)))


def test_Divide_mixed_consts():
    transformer = PropagateConstants()

    expr = Divide(Symbol("a"), Constant(-2))
    new_expr = transformer.visit(expr)
    assert new_expr is not expr
    assert isinstance(new_expr, Divide)
    assert new_expr.expr1 is Symbol("a")
    assert new_expr.expr2 is Constant(-2)

    expr = Divide(Constant(10), Symbol("b"))
    new_expr = transformer.visit(expr)
    assert new_expr is not expr
    assert isinstance(new_expr, Divide)
    assert new_expr.expr1 is Constant(10)
    assert new_expr.expr2 is Symbol("b")


def test_Divide_by_zero():
    transformer = PropagateConstants()

    expr = Divide(Symbol("a"), Constant(0))
    with pytest.raises(ZeroDivisionError, match="(a / 0)"):
        new_expr = transformer.visit(expr)

    expr = Divide(Symbol("a"), Constant(0.0))
    with pytest.raises(ZeroDivisionError, match="(a / 0)"):
        new_expr = transformer.visit(expr)


def test_Divide_by_one():
    transformer = PropagateConstants()

    expr = Divide(Symbol("a"), Constant(1))
    new_expr = transformer.visit(expr)
    assert new_expr is not expr
    assert new_expr is Symbol("a")

    expr = Divide(Symbol("a"), Constant(1.0))
    new_expr = transformer.visit(expr)
    assert new_expr is not expr
    assert new_expr is Symbol("a")


def test_Divide_by_negative_one():
    transformer = PropagateConstants()

    expr = Divide(Symbol("a"), Constant(-1))
    new_expr = transformer.visit(expr)
    assert new_expr is not expr
    assert isinstance(new_expr, Negation)
    assert new_expr.expr is Symbol("a")

    expr = Divide(Symbol("a"), Constant(-1.0))
    new_expr = transformer.visit(expr)
    assert new_expr is not expr
    assert isinstance(new_expr, Negation)
    assert new_expr.expr is Symbol("a")


def test_Divide_zero():
    transformer = PropagateConstants()

    expr = Divide(Constant(0), Symbol("b"))
    new_expr = transformer.visit(expr)
    assert new_expr is not expr
    assert new_expr is Constant(0)

    expr = Divide(Constant(0.0), Symbol("b"))
    new_expr = transformer.visit(expr)
    assert new_expr is not expr
    assert new_expr is Constant(0.0)
