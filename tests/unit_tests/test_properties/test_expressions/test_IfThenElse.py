import pytest

from dnnv.properties.expressions import *


def test_IfThenElse():
    expr = IfThenElse(Symbol("a"), Symbol("b"), Symbol("c"))
    assert expr.expr1 == Symbol("a")
    assert expr.expr2 == Symbol("b")
    assert expr.expr3 == Symbol("c")

    expr = IfThenElse(Constant(True), Constant(1), Constant(-1))
    assert expr.expr1 == Constant(True)
    assert expr.expr2 == Constant(1)
    assert expr.expr3 == Constant(-1)


def test_condition():
    expr = IfThenElse(Symbol("a"), Symbol("b"), Symbol("c"))
    assert expr.condition is expr.expr1

    expr = IfThenElse(Constant(True), Constant(1), Constant(-1))
    assert expr.condition is expr.expr1


def test_t_expr():
    expr = IfThenElse(Symbol("a"), Symbol("b"), Symbol("c"))
    assert expr.t_expr is expr.expr2

    expr = IfThenElse(Constant(True), Constant(1), Constant(-1))
    assert expr.t_expr is expr.expr2


def test_f_expr():
    expr = IfThenElse(Symbol("a"), Symbol("b"), Symbol("c"))
    assert expr.f_expr is expr.expr3

    expr = IfThenElse(Constant(True), Constant(1), Constant(-1))
    assert expr.f_expr is expr.expr3


def test_value():
    expr = IfThenElse(Symbol("a"), Symbol("b"), Symbol("c"))
    with pytest.raises(ValueError):
        _ = expr.value

    expr = IfThenElse(Constant(True), Constant(1), Constant(-1))
    assert expr.value == 1

    expr = IfThenElse(Constant(False), Constant(1), Constant(-1))
    assert expr.value == -1
