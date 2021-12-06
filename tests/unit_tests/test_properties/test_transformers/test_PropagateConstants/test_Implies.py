import numpy as np
import pytest

from dnnv.properties.expressions import *
from dnnv.properties.transformers import PropagateConstants


def test_Implies_symbols():
    transformer = PropagateConstants()

    a, b = Symbol("a"), Symbol("b")
    expr = Implies(a, b)
    new_expr = transformer.visit(expr)
    assert new_expr is not expr
    assert isinstance(new_expr, Implies)
    assert new_expr.expr1 is Symbol("a")
    assert new_expr.expr2 is Symbol("b")


@pytest.mark.xfail
def test_Implies_same_symbol():
    transformer = PropagateConstants()

    a, b = Symbol("x"), Symbol("x")
    expr = Implies(a, b)
    new_expr = transformer.visit(expr)
    assert new_expr is not expr
    assert isinstance(new_expr, Constant)
    assert new_expr.value == True


def test_Implies_consts():
    transformer = PropagateConstants()

    a, b = Constant(True), Constant(True)
    expr = Implies(a, b)
    new_expr = transformer.visit(expr)
    assert new_expr is not expr
    assert isinstance(new_expr, Constant)
    assert new_expr.value == True

    a, b = Constant(True), Constant(False)
    expr = Implies(a, b)
    new_expr = transformer.visit(expr)
    assert new_expr is not expr
    assert isinstance(new_expr, Constant)
    assert new_expr.value == False

    a, b = Constant(False), Constant(True)
    expr = Implies(a, b)
    new_expr = transformer.visit(expr)
    assert new_expr is not expr
    assert isinstance(new_expr, Constant)
    assert new_expr.value == True

    a, b = Constant(False), Constant(False)
    expr = Implies(a, b)
    new_expr = transformer.visit(expr)
    assert new_expr is not expr
    assert isinstance(new_expr, Constant)
    assert new_expr.value == True


def test_Implies_mixed():
    transformer = PropagateConstants()

    a, b = Symbol("a"), Constant(True)
    expr = Implies(a, b)
    new_expr = transformer.visit(expr)
    assert new_expr is not expr
    assert isinstance(new_expr, Constant)
    assert new_expr.value == True

    a, b = Symbol("a"), Constant(False)
    expr = Implies(a, b)
    new_expr = transformer.visit(expr)
    assert new_expr is not expr
    assert isinstance(new_expr, Not)
    assert new_expr.expr is Symbol("a")


def test_Implies_const_arrays():
    transformer = PropagateConstants()

    a, b = Constant(np.ones((1, 3), dtype=bool)), Constant(np.ones((1, 3), dtype=bool))
    expr = Implies(a, b)
    new_expr = transformer.visit(expr)
    assert new_expr is not expr
    assert isinstance(new_expr, Constant)
    assert new_expr.value == True

    a, b = Constant(np.ones((1, 3), dtype=bool)), Constant(np.zeros((1, 3), dtype=bool))
    expr = Implies(a, b)
    new_expr = transformer.visit(expr)
    assert new_expr is not expr
    assert isinstance(new_expr, Constant)
    assert new_expr.value == False

    a, b = Constant(np.zeros((1, 3), dtype=bool)), Constant(np.ones((1, 3), dtype=bool))
    expr = Implies(a, b)
    new_expr = transformer.visit(expr)
    assert new_expr is not expr
    assert isinstance(new_expr, Constant)
    assert new_expr.value == True

    a, b = Constant(np.zeros((1, 3), dtype=bool)), Constant(
        np.zeros((1, 3), dtype=bool)
    )
    expr = Implies(a, b)
    new_expr = transformer.visit(expr)
    assert new_expr is not expr
    assert isinstance(new_expr, Constant)
    assert new_expr.value == True

    arr_a = np.array([1, 0, 1, 0], dtype=bool)
    arr_b = np.array([1, 0, 0, 1], dtype=bool)
    a, b = Constant(arr_a), Constant(arr_b)
    expr = Implies(a, b)
    new_expr = transformer.visit(expr)
    assert new_expr is not expr
    assert isinstance(new_expr, Constant)
    assert np.all(new_expr.value == np.array([1, 1, 0, 1], dtype=bool))


def test_Implies_array_for_antecedent():
    transformer = PropagateConstants()

    a, b = Constant(np.ones((1, 3), dtype=bool)), Symbol("b")
    expr = Implies(a, b)
    new_expr = transformer.visit(expr)
    assert new_expr is not expr
    assert new_expr is Symbol("b")

    a, b = Constant(np.zeros((1, 3), dtype=bool)), Symbol("b")
    expr = Implies(a, b)
    new_expr = transformer.visit(expr)
    assert new_expr is not expr
    assert isinstance(new_expr, Constant)
    assert new_expr.value == True

    arr_a = np.array([1, 0], dtype=bool)
    a, b = Constant(arr_a), Symbol("b")
    expr = Implies(a, b)
    new_expr = transformer.visit(expr)
    assert new_expr is not expr
    assert isinstance(new_expr, Implies)
    assert np.allclose(new_expr.expr1.value, arr_a)
    assert new_expr.expr2 is Symbol("b")


def test_Implies_array_for_consequent():
    transformer = PropagateConstants()

    a, b = Symbol("a"), Constant(np.zeros((1, 3), dtype=bool))
    expr = Implies(a, b)
    new_expr = transformer.visit(expr)
    assert new_expr is not expr
    assert isinstance(new_expr, Not)
    assert new_expr.expr is Symbol("a")

    a, b = Symbol("a"), Constant(np.ones((1, 3), dtype=bool))
    expr = Implies(a, b)
    new_expr = transformer.visit(expr)
    assert new_expr is not expr
    assert isinstance(new_expr, Constant)
    assert new_expr.value == True

    arr_b = np.array([1, 0], dtype=bool)
    a, b = Symbol("a"), Constant(arr_b)
    expr = Implies(a, b)
    new_expr = transformer.visit(expr)
    assert new_expr is not expr
    assert isinstance(new_expr, Implies)
    assert new_expr.expr1 is Symbol("a")
    assert np.allclose(new_expr.expr2.value, arr_b)
