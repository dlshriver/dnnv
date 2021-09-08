import pytest

from dnnv.properties.base import *
from dnnv.properties.transformers import Canonical


def test_LessThanOrEqual_symbols():
    transformer = Canonical()

    expr = LessThanOrEqual(Symbol("a"), Symbol("b"))
    new_expr = transformer.visit(expr)
    assert new_expr is not expr
    assert isinstance(new_expr, And)
    assert len(new_expr.expressions) == 1
    assert isinstance(new_expr.expressions[0], Or)
    assert len(new_expr.expressions[0].expressions) == 1
    assert isinstance(new_expr.expressions[0].expressions[0], GreaterThanOrEqual)
    new_expr_gt = new_expr.expressions[0].expressions[0]
    assert isinstance(new_expr_gt.expr1, Add)
    assert len(new_expr_gt.expr1.expressions) == 2
    assert Multiply(Constant(-1), Symbol("a")) in new_expr_gt.expr1.expressions
    assert Multiply(Constant(1), Symbol("b")) in new_expr_gt.expr1.expressions
    assert isinstance(new_expr_gt.expr2, Constant)
    assert new_expr_gt.expr2 is Constant(0)


def test_LessThanOrEqual_constants():
    transformer = Canonical()

    expr = LessThanOrEqual(Constant(9), Constant(5))
    new_expr = transformer.visit_LessThanOrEqual(expr)
    assert new_expr is not expr
    assert isinstance(new_expr, GreaterThanOrEqual)
    new_expr_gt = new_expr
    assert isinstance(new_expr_gt.expr1, Add)
    assert len(new_expr_gt.expr1.expressions) == 0
    assert isinstance(new_expr_gt.expr2, Constant)
    assert new_expr_gt.expr2 is Constant(4)


def test_LessThanOrEqual_mixed_symbols_constants_0():
    transformer = Canonical()

    a, b = Symbol("a"), Symbol("b")
    expr = LessThanOrEqual(a + b, Constant(5))
    new_expr = transformer.visit(expr)
    assert new_expr is not expr
    assert isinstance(new_expr, And)
    assert len(new_expr.expressions) == 1
    assert isinstance(new_expr.expressions[0], Or)
    assert len(new_expr.expressions[0].expressions) == 1
    assert isinstance(new_expr.expressions[0].expressions[0], GreaterThanOrEqual)
    new_expr_gt = new_expr.expressions[0].expressions[0]
    assert isinstance(new_expr_gt.expr1, Add)
    assert len(new_expr_gt.expr1.expressions) == 2
    assert Multiply(Constant(-1), Symbol("a")) in new_expr_gt.expr1.expressions
    assert Multiply(Constant(-1), Symbol("b")) in new_expr_gt.expr1.expressions
    assert isinstance(new_expr_gt.expr2, Constant)
    assert new_expr_gt.expr2 is Constant(-5)


def test_LessThanOrEqual_mixed_symbols_constants_1():
    transformer = Canonical()

    a, b = Symbol("a"), Symbol("b")
    expr = LessThanOrEqual(Constant(5), a + b)
    new_expr = transformer.visit(expr)
    assert new_expr is not expr
    assert isinstance(new_expr, And)
    assert len(new_expr.expressions) == 1
    assert isinstance(new_expr.expressions[0], Or)
    assert len(new_expr.expressions[0].expressions) == 1
    assert isinstance(new_expr.expressions[0].expressions[0], GreaterThanOrEqual)
    new_expr_gt = new_expr.expressions[0].expressions[0]
    assert isinstance(new_expr_gt.expr1, Add)
    assert len(new_expr_gt.expr1.expressions) == 2
    assert Multiply(Constant(1), Symbol("a")) in new_expr_gt.expr1.expressions
    assert Multiply(Constant(1), Symbol("b")) in new_expr_gt.expr1.expressions
    assert isinstance(new_expr_gt.expr2, Constant)
    assert new_expr_gt.expr2 is Constant(5)