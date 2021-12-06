import numpy as np

from dnnv.properties.expressions import *
from dnnv.properties.transformers import PropagateConstants


def test_Add_symbols():
    transformer = PropagateConstants()

    a, b, c = Symbol("a"), Symbol("b"), Symbol("c")
    expr = a + b + c
    new_expr = transformer.visit(expr)
    assert new_expr is not expr
    assert isinstance(new_expr, Add)
    assert Symbol("a") in new_expr.expressions
    assert Symbol("b") in new_expr.expressions
    assert Symbol("c") in new_expr.expressions


def test_Add_consts():
    transformer = PropagateConstants()

    a, b, c = Constant("a"), Constant("b"), Constant("c")
    expr = a + b + c
    new_expr = transformer.visit(expr)
    assert new_expr is not expr
    assert isinstance(new_expr, Constant)
    assert new_expr.value == "abc"


def test_Add_const_arrays():
    transformer = PropagateConstants()

    a, b, c = Constant(np.ones((1, 3))), Constant(3), Constant(-2)
    expr = a + b + c
    new_expr = transformer.visit(expr)
    assert new_expr is not expr
    assert isinstance(new_expr, Constant)
    assert np.allclose(new_expr.value, (np.ones((1, 3)) + 1))
    assert np.allclose(a.value, np.ones((1, 3)))

    a, b, c = Constant(np.ones((1, 3))), Constant(3), Constant(np.ones((1, 3)))
    expr = a + b + c
    new_expr = transformer.visit(expr)
    assert new_expr is not expr
    assert isinstance(new_expr, Constant)
    assert np.allclose(new_expr.value, (np.ones((1, 3)) + 4))
    assert np.allclose(a.value, np.ones((1, 3)))
    assert np.allclose(c.value, np.ones((1, 3)))

    a, b, c = Constant(-4), Constant(3), Constant(np.ones((1, 3)))
    expr = a + b + c
    new_expr = transformer.visit(expr)
    assert new_expr is not expr
    assert isinstance(new_expr, Constant)
    assert np.allclose(new_expr.value, (np.zeros((1, 3))))
    assert np.allclose(c.value, np.ones((1, 3)))


def test_Add_empty():
    transformer = PropagateConstants()

    expr = Add()
    new_expr = transformer.visit(expr)
    assert new_expr is not expr
    assert isinstance(new_expr, Constant)
    assert new_expr.value == 0


def test_Add_single_nonconst_expr():
    transformer = PropagateConstants()

    expr = Add(Symbol("a"))
    new_expr = transformer.visit(expr)
    assert new_expr is not expr
    assert new_expr is Symbol("a")


def test_Add_mixed_consts_eq_zero():
    transformer = PropagateConstants()

    expr = Add(Symbol("a"), Constant(-1), Constant(-3), Symbol("b"), Constant(4))
    new_expr = transformer.visit(expr)
    assert new_expr is not expr
    assert isinstance(new_expr, Add)
    assert len(new_expr.expressions) == 2
    assert Symbol("a") in new_expr.expressions
    assert Symbol("b") in new_expr.expressions


def test_Add_mixed_consts_neq_zero():
    transformer = PropagateConstants()

    expr = Add(Symbol("a"), Constant(-1), Symbol("b"), Constant(4))
    new_expr = transformer.visit(expr)
    assert new_expr is not expr
    assert isinstance(new_expr, Add)
    assert len(new_expr.expressions) == 3
    assert Symbol("a") in new_expr.expressions
    assert Symbol("b") in new_expr.expressions
    assert Constant(3) in new_expr.expressions
