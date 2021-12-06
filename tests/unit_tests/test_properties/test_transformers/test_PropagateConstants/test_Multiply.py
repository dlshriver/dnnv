import numpy as np

from dnnv.properties.expressions import *
from dnnv.properties.transformers import PropagateConstants


def test_Multiply_symbols():
    transformer = PropagateConstants()

    a, b, c = Symbol("a"), Symbol("b"), Symbol("c")
    expr = Multiply(a, b, c)
    new_expr = transformer.visit(expr)
    assert new_expr is not expr
    assert isinstance(new_expr, Multiply)
    assert Symbol("a") in new_expr.expressions
    assert Symbol("b") in new_expr.expressions
    assert Symbol("c") in new_expr.expressions


def test_Multiply_consts():
    transformer = PropagateConstants()

    a, b, c = Constant(2), Constant(3), Constant(-5)
    expr = Multiply(a, b, c)
    new_expr = transformer.visit(expr)
    assert new_expr is not expr
    assert isinstance(new_expr, Constant)
    assert new_expr.value == -30


def test_Multiply_const_arrays():
    transformer = PropagateConstants()

    a, b, c = Constant(np.ones((1, 3))), Constant(3), Constant(-2)
    expr = Multiply(a, b, c)
    new_expr = transformer.visit(expr)
    assert new_expr is not expr
    assert isinstance(new_expr, Constant)
    assert np.allclose(new_expr.value, np.full((1, 3), -6))
    assert np.allclose(a.value, np.ones((1, 3)))

    a, b, c = Constant(np.ones((1, 3))), Constant(3), Constant(np.ones((1, 3)))
    expr = Multiply(a, b, c)
    new_expr = transformer.visit(expr)
    assert new_expr is not expr
    assert isinstance(new_expr, Constant)
    assert np.allclose(new_expr.value, np.full((1, 3), 3))
    assert np.allclose(a.value, np.ones((1, 3)))
    assert np.allclose(c.value, np.ones((1, 3)))

    a, b, c = Constant(-4), Constant(3), Constant(np.ones((1, 3)))
    expr = Multiply(a, b, c)
    new_expr = transformer.visit(expr)
    assert new_expr is not expr
    assert isinstance(new_expr, Constant)
    assert np.allclose(new_expr.value, np.full((1, 3), -12))
    assert np.allclose(c.value, np.ones((1, 3)))


def test_Multiply_empty():
    transformer = PropagateConstants()

    expr = Multiply()
    new_expr = transformer.visit(expr)
    assert new_expr is not expr
    assert isinstance(new_expr, Constant)
    assert new_expr.value == 1


def test_Multiply_single_nonconst_expr():
    transformer = PropagateConstants()

    expr = Multiply(Symbol("a"))
    new_expr = transformer.visit(expr)
    assert new_expr is not expr
    assert new_expr is Symbol("a")


def test_Multiply_mixed_consts_with_zero():
    transformer = PropagateConstants()

    expr = Multiply(Symbol("a"), Constant(-1), Constant(0), Symbol("b"), Constant(-1))
    new_expr = transformer.visit(expr)
    assert new_expr is not expr
    assert isinstance(new_expr, Constant)
    assert new_expr.value == 0


def test_Multiply_mixed_consts_eq_one():
    transformer = PropagateConstants()

    expr = Multiply(Symbol("a"), Constant(-1), Constant(1), Symbol("b"), Constant(-1))
    new_expr = transformer.visit(expr)
    assert new_expr is not expr
    assert isinstance(new_expr, Multiply)
    assert len(new_expr.expressions) == 2
    assert Symbol("a") in new_expr.expressions
    assert Symbol("b") in new_expr.expressions

    expr = Multiply(Symbol("a"), Constant(1))
    new_expr = transformer.visit(expr)
    assert new_expr is not expr
    assert new_expr is Symbol("a")


def test_Multiply_mixed_consts_neq_one():
    transformer = PropagateConstants()

    expr = Multiply(Symbol("a"), Constant(-1), Symbol("b"), Constant(4))
    new_expr = transformer.visit(expr)
    assert new_expr is not expr
    assert isinstance(new_expr, Multiply)
    assert len(new_expr.expressions) == 3
    assert Symbol("a") in new_expr.expressions
    assert Symbol("b") in new_expr.expressions
    assert Constant(-4) in new_expr.expressions
