import numpy as np

from dnnv.properties.expressions import *
from dnnv.properties.transformers import PropagateConstants


def test_Subtract_symbols():
    transformer = PropagateConstants()

    a, b = Symbol("a"), Symbol("b")
    expr = Subtract(a, b)
    new_expr = transformer.visit(expr)
    assert new_expr is not expr
    assert isinstance(new_expr, Subtract)
    assert new_expr.expr1 is Symbol("a")
    assert new_expr.expr2 is Symbol("b")


def test_Subtract_consts():
    transformer = PropagateConstants()

    a, b = Constant(6), Constant(2)
    expr = Subtract(a, b)
    new_expr = transformer.visit(expr)
    assert new_expr is not expr
    assert isinstance(new_expr, Constant)
    assert new_expr.value == 4


def test_Subtract_const_arrays():
    transformer = PropagateConstants()

    arr = np.full((1, 3), 4.0)
    a, b = Constant(arr), Constant(2)
    expr = Subtract(a, b)
    new_expr = transformer.visit(expr)
    assert new_expr is not expr
    assert isinstance(new_expr, Constant)
    assert np.allclose(new_expr.value, arr - 2)
    assert np.allclose(a.value, arr)

    arr = np.full((1, 3), 3.0)
    a, b = Constant(arr), Constant(np.ones((1, 3)))
    expr = Subtract(a, b)
    new_expr = transformer.visit(expr)
    assert new_expr is not expr
    assert isinstance(new_expr, Constant)
    assert np.allclose(new_expr.value, np.full((1, 3), 2.0))
    assert np.allclose(a.value, arr)
    assert np.allclose(b.value, np.ones((1, 3)))

    arr = np.full((1, 3), 2.0)
    a, b = Constant(3), Constant(np.ones((1, 3)))
    expr = Subtract(a, b)
    new_expr = transformer.visit(expr)
    assert new_expr is not expr
    assert isinstance(new_expr, Constant)
    assert np.allclose(new_expr.value, arr)
    assert np.allclose(b.value, np.ones((1, 3)))


def test_Subtract_self():
    transformer = PropagateConstants()

    expr = Subtract(Symbol("a"), Symbol("a"))
    new_expr = transformer.visit(expr)
    assert new_expr is not expr
    assert isinstance(new_expr, Constant)
    assert new_expr.value == 0


def test_Subtract_zero():
    transformer = PropagateConstants()

    expr = Subtract(Symbol("a"), Constant(0))
    new_expr = transformer.visit(expr)
    assert new_expr is not expr
    assert new_expr is Symbol("a")

    expr = Subtract(Symbol("a"), Constant(0.0))
    new_expr = transformer.visit(expr)
    assert new_expr is not expr
    assert new_expr is Symbol("a")


def test_Subtract_from_zero():
    transformer = PropagateConstants()

    expr = Subtract(Constant(0), Symbol("b"))
    new_expr = transformer.visit(expr)
    assert new_expr is not expr
    assert isinstance(new_expr, Negation)
    assert new_expr.expr is Symbol("b")

    expr = Subtract(Constant(0.0), Symbol("b"))
    new_expr = transformer.visit(expr)
    assert new_expr is not expr
    assert isinstance(new_expr, Negation)
    assert new_expr.expr is Symbol("b")
