import pytest

from dnnv.properties.expressions import *


def test_Slice():
    expr = Slice(Symbol("a"), Symbol("b"), Symbol("c"))
    assert expr.expr1 == Symbol("a")
    assert expr.expr2 == Symbol("b")
    assert expr.expr3 == Symbol("c")

    expr = Slice(0, -1, 1)
    assert expr.expr1 == Constant(0)
    assert expr.expr2 == Constant(-1)
    assert expr.expr3 == Constant(1)


def test_start():
    expr = Slice(Symbol("a"), Symbol("b"), Symbol("c"))
    assert expr.start is expr.expr1

    expr = Slice(0, -1, 1)
    assert expr.start is expr.expr1


def test_stop():
    expr = Slice(Symbol("a"), Symbol("b"), Symbol("c"))
    assert expr.stop is expr.expr2

    expr = Slice(0, -1, 1)
    assert expr.stop is expr.expr2


def test_step():
    expr = Slice(Symbol("a"), Symbol("b"), Symbol("c"))
    assert expr.step is expr.expr3

    expr = Slice(0, -1, 1)
    assert expr.step is expr.expr3


def test_value():
    expr = Slice(Symbol("a"), Symbol("b"), Symbol("c"))
    with pytest.raises(ValueError):
        _ = expr.value

    expr = Slice(0, -1, 1)
    assert expr.value == slice(0, -1, 1)


def test_repr():
    expr = Slice(0, -1, 1)
    assert repr(expr) == "0:-1:1"

    expr = Slice(0, -1, None)
    assert repr(expr) == "0:-1"


def test_str():
    expr = Slice(0, -1, 1)
    assert str(expr) == "0:-1:1"
    expr = Slice(0, -1, None)
    assert str(expr) == "0:-1"


def test_ExtSlice():
    expr = ExtSlice(Symbol("a"), Symbol("b"), Symbol("c"))
    assert Symbol("a") in expr.expressions
    assert Symbol("b") in expr.expressions
    assert Symbol("c") in expr.expressions

    expr = ExtSlice(Constant(0), Constant(1), Constant(2))
    assert Constant(0) in expr.expressions
    assert Constant(1) in expr.expressions
    assert Constant(2) in expr.expressions


def test_ExtSlice_value():
    expr = ExtSlice(Symbol("a"), Symbol("b"), Symbol("c"))
    with pytest.raises(ValueError):
        _ = expr.value

    expr = ExtSlice()
    assert expr.value == ()

    expr = ExtSlice(Constant(0), Constant(1), Constant(2))
    assert expr.value == (0, 1, 2)
