import numpy as np
import pytest

from dnnv.properties.expressions import *
from dnnv.properties.transformers import PropagateConstants


def test_Or_symbols():
    transformer = PropagateConstants()

    a, b, c = Symbol("a"), Symbol("b"), Symbol("c")
    expr = Or(a, b, c)
    new_expr = transformer.visit(expr)
    assert new_expr is not expr
    assert isinstance(new_expr, Or)
    assert Symbol("a") in new_expr.expressions
    assert Symbol("b") in new_expr.expressions
    assert Symbol("c") in new_expr.expressions


def test_Or_consts():
    transformer = PropagateConstants()

    a, b, c = Constant(True), Constant(False), Constant(True)
    expr = Or(a, b, c)
    new_expr = transformer.visit(expr)
    assert new_expr is not expr
    assert isinstance(new_expr, Constant)
    assert new_expr.value == True

    a, b, c = Constant(True), Constant(True), Constant(True)
    expr = Or(a, b, c)
    new_expr = transformer.visit(expr)
    assert new_expr is not expr
    assert isinstance(new_expr, Constant)
    assert new_expr.value == True

    a, b, c = Constant(False), Constant(False), Constant(False)
    expr = Or(a, b, c)
    new_expr = transformer.visit(expr)
    assert new_expr is not expr
    assert isinstance(new_expr, Constant)
    assert new_expr.value == False


def test_Or_const_arrays():
    transformer = PropagateConstants()

    a, b, c = Constant(np.ones((1, 3), dtype=bool)), Constant(True), Constant(False)
    expr = Or(a, b, c)
    new_expr = transformer.visit(expr)
    assert new_expr is not expr
    assert isinstance(new_expr, Constant)
    assert np.allclose(new_expr.value, np.ones((1, 3), dtype=bool))
    assert np.allclose(a.value, np.ones((1, 3), dtype=bool))

    a, b, c = (
        Constant(np.ones((1, 3), dtype=bool)),
        Constant(False),
        Constant(np.ones((1, 3), dtype=bool)),
    )
    expr = Or(a, b, c)
    new_expr = transformer.visit(expr)
    assert new_expr is not expr
    assert isinstance(new_expr, Constant)
    assert np.allclose(new_expr.value, np.ones((1, 3), dtype=bool))
    assert np.allclose(a.value, np.ones((1, 3), dtype=bool))
    assert np.allclose(c.value, np.ones((1, 3), dtype=bool))

    a, b, c = Constant(True), Constant(True), Constant(np.zeros((1, 3), dtype=bool))
    expr = Or(a, b, c)
    new_expr = transformer.visit(expr)
    assert new_expr is not expr
    assert isinstance(new_expr, Constant)
    assert np.allclose(new_expr.value, np.ones((1, 3), dtype=bool))
    assert np.allclose(c.value, np.zeros((1, 3)))

    a, b, c = (
        Constant(False),
        Constant(False),
        Constant(np.zeros((1, 3), dtype=bool)),
    )
    expr = Or(a, b, c)
    new_expr = transformer.visit(expr)
    assert new_expr is not expr
    assert isinstance(new_expr, Constant)
    assert np.allclose(new_expr.value, np.zeros((1, 3), dtype=bool))
    assert np.allclose(c.value, np.zeros((1, 3)))


def test_Or_empty():
    transformer = PropagateConstants()

    expr = Or()
    new_expr = transformer.visit(expr)
    assert new_expr is not expr
    assert isinstance(new_expr, Constant)
    assert new_expr.value == False


def test_Or_single_nonconst_expr():
    transformer = PropagateConstants()

    expr = Or(Symbol("a"))
    new_expr = transformer.visit(expr)
    assert new_expr is not expr
    assert new_expr is Symbol("a")


def test_Or_mixed_consts_eq_false():
    transformer = PropagateConstants()

    expr = Or(Symbol("a"), Constant(False), Symbol("b"), Constant(True))
    new_expr = transformer.visit(expr)
    assert new_expr is not expr
    assert isinstance(new_expr, Constant)
    assert new_expr.value == True


def test_Or_mixed_consts_neq_false():
    transformer = PropagateConstants()

    expr = Or(Symbol("a"), Constant(False), Symbol("b"), Constant(False))
    new_expr = transformer.visit(expr)
    assert new_expr is not expr
    assert isinstance(new_expr, Or)
    assert len(new_expr.expressions) == 2
    assert Symbol("a") in new_expr.expressions
    assert Symbol("b") in new_expr.expressions


def test_Or_mixed_const_arrays():
    transformer = PropagateConstants()

    expr = Or(
        Symbol("a"),
        Constant(np.ones((1, 3), dtype=bool)),
        Symbol("b"),
        Constant(np.ones((1, 3), dtype=bool)),
    )
    new_expr = transformer.visit(expr)
    assert new_expr is not expr
    assert isinstance(new_expr, Constant)
    assert np.allclose(new_expr.value, np.ones((1, 3), dtype=bool))

    expr = Or(
        Symbol("a"),
        Constant(np.zeros((1, 3), dtype=bool)),
        Symbol("b"),
    )
    new_expr = transformer.visit(expr)
    assert new_expr is not expr
    assert isinstance(new_expr, Or)
    assert len(new_expr.expressions) == 2
    assert Symbol("a") in new_expr.expressions
    assert Symbol("b") in new_expr.expressions

    expr = Or(
        Symbol("a"),
        Constant(np.zeros((1, 3), dtype=bool)),
    )
    new_expr = transformer.visit(expr)
    assert new_expr is not expr
    assert new_expr is Symbol("a")

    arr = np.array([0, 1, 0], dtype=bool)
    expr = Or(
        Symbol("a"),
        Constant(arr),
        Symbol("b"),
    )
    new_expr = transformer.visit(expr)
    assert new_expr is not expr
    assert isinstance(new_expr, Or)
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
def test_Or_symbol_and_neg_symbol():
    transformer = PropagateConstants()

    expr = Or(Symbol("a"), Symbol("b"), ~Symbol("a"))
    new_expr = transformer.visit(expr)
    assert new_expr is not expr
    assert isinstance(new_expr, Constant)
    assert new_expr.value == True
