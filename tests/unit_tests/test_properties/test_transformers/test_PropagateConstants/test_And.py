import numpy as np
import pytest

from dnnv.properties.expressions import *
from dnnv.properties.transformers import PropagateConstants


def test_And_symbols():
    transformer = PropagateConstants()

    a, b, c = Symbol("a"), Symbol("b"), Symbol("c")
    expr = a & b & c
    new_expr = transformer.visit(expr)
    assert new_expr is not expr
    assert isinstance(new_expr, And)
    assert Symbol("a") in new_expr.expressions
    assert Symbol("b") in new_expr.expressions
    assert Symbol("c") in new_expr.expressions


def test_And_consts():
    transformer = PropagateConstants()

    a, b, c = Constant(True), Constant(False), Constant(True)
    expr = a & b & c
    new_expr = transformer.visit(expr)
    assert new_expr is not expr
    assert isinstance(new_expr, Constant)
    assert new_expr.value == False

    a, b, c = Constant(True), Constant(True), Constant(True)
    expr = a & b & c
    new_expr = transformer.visit(expr)
    assert new_expr is not expr
    assert isinstance(new_expr, Constant)
    assert new_expr.value == True


def test_And_const_arrays():
    transformer = PropagateConstants()

    a, b, c = Constant(np.ones((1, 3), dtype=bool)), Constant(True), Constant(False)
    expr = a & b & c
    new_expr = transformer.visit(expr)
    assert new_expr is not expr
    assert isinstance(new_expr, Constant)
    assert np.allclose(new_expr.value, np.zeros((1, 3), dtype=bool))
    assert np.allclose(a.value, np.ones((1, 3), dtype=bool))

    a, b, c = (
        Constant(np.ones((1, 3), dtype=bool)),
        Constant(False),
        Constant(np.ones((1, 3), dtype=bool)),
    )
    expr = a & b & c
    new_expr = transformer.visit(expr)
    assert new_expr is not expr
    assert isinstance(new_expr, Constant)
    assert np.allclose(new_expr.value, np.zeros((1, 3), dtype=bool))
    assert np.allclose(a.value, np.ones((1, 3), dtype=bool))
    assert np.allclose(c.value, np.ones((1, 3), dtype=bool))

    a, b, c = Constant(True), Constant(True), Constant(np.zeros((1, 3), dtype=bool))
    expr = a & b & c
    new_expr = transformer.visit(expr)
    assert new_expr is not expr
    assert isinstance(new_expr, Constant)
    assert np.allclose(new_expr.value, np.zeros((1, 3), dtype=bool))
    assert np.allclose(c.value, np.zeros((1, 3)))


def test_And_empty():
    transformer = PropagateConstants()

    expr = And()
    new_expr = transformer.visit(expr)
    assert new_expr is not expr
    assert isinstance(new_expr, Constant)
    assert new_expr.value == True


def test_And_single_nonconst_expr():
    transformer = PropagateConstants()

    expr = And(Symbol("a"))
    new_expr = transformer.visit(expr)
    assert new_expr is not expr
    assert new_expr is Symbol("a")


def test_And_mixed_consts_eq_false():
    transformer = PropagateConstants()

    expr = And(Symbol("a"), Constant(False), Symbol("b"), Constant(True))
    new_expr = transformer.visit(expr)
    assert new_expr is not expr
    assert isinstance(new_expr, Constant)
    assert new_expr.value == False


def test_And_mixed_consts_neq_false():
    transformer = PropagateConstants()

    expr = And(Symbol("a"), Constant(True), Symbol("b"), Constant(True))
    new_expr = transformer.visit(expr)
    assert new_expr is not expr
    assert isinstance(new_expr, And)
    assert len(new_expr.expressions) == 2
    assert Symbol("a") in new_expr.expressions
    assert Symbol("b") in new_expr.expressions


def test_And_mixed_const_arrays():
    transformer = PropagateConstants()

    expr = And(
        Symbol("a"),
        Constant(np.ones((1, 3), dtype=bool)),
        Symbol("b"),
        Constant(np.ones((1, 3), dtype=bool)),
    )
    new_expr = transformer.visit(expr)
    assert new_expr is not expr
    assert isinstance(new_expr, And)
    assert len(new_expr.expressions) == 2
    assert Symbol("a") in new_expr.expressions
    assert Symbol("b") in new_expr.expressions

    expr = And(
        Symbol("a"),
        Constant(np.ones((1, 3), dtype=bool)),
        Constant(np.ones((1, 3), dtype=bool)),
    )
    new_expr = transformer.visit(expr)
    assert new_expr is not expr
    assert new_expr is Symbol("a")

    expr = And(
        Symbol("a"),
        Constant(np.ones((1, 3), dtype=bool)),
        Symbol("b"),
        Constant(np.zeros((1, 3), dtype=bool)),
    )
    new_expr = transformer.visit(expr)
    assert new_expr is not expr
    assert isinstance(new_expr, Constant)
    assert np.all(new_expr.value == False)

    arr = np.ones((1, 3), dtype=bool)
    arr[0, 1] = False
    expr = And(
        Symbol("a"),
        Constant(arr),
        Symbol("b"),
        Constant(np.ones((1, 3), dtype=bool)),
    )
    new_expr = transformer.visit(expr)
    assert new_expr is not expr
    assert isinstance(new_expr, And)
    assert len(new_expr.expressions) == 3
    assert Symbol("a") in new_expr.expressions
    assert Symbol("b") in new_expr.expressions
    for e in new_expr.expressions:
        if isinstance(e, Constant):
            assert np.allclose(e.value, arr)
            break
    else:
        assert False


@pytest.mark.xfail
def test_And_symbol_and_neg_symbol():
    transformer = PropagateConstants()

    expr = And(Symbol("a"), Symbol("b"), ~Symbol("a"))
    new_expr = transformer.visit(expr)
    assert new_expr is not expr
    assert isinstance(new_expr, Constant)
    assert new_expr.value == False
