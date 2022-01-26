import numpy as np

from dnnv.properties import *
from dnnv.properties.transformers import DnfTransformer


def test_Not_Symbol():
    transformer = DnfTransformer()

    expr = Not(Symbol("x"))
    new_expr = transformer.visit(expr)
    assert new_expr is not expr
    assert isinstance(new_expr, Or)
    assert len(new_expr.expressions) == 1
    assert And(Not(Symbol("x"))) in new_expr.expressions


def test_Not_Implies():
    transformer = DnfTransformer()

    a, c = Symbol("a"), Symbol("c")
    expr = Not(Implies(a, c))
    new_expr = transformer.visit(expr)
    assert new_expr is not expr
    assert isinstance(new_expr, Or)
    assert len(new_expr.expressions) == 1
    assert And(a, Not(c)) in new_expr.expressions


def test_Not_LessThan_non_concrete_shapes():
    transformer = DnfTransformer()

    expr = Not(LessThan(Symbol("x"), Symbol("y")))
    new_expr = transformer.visit(expr)
    assert new_expr is not expr
    assert isinstance(new_expr, Or)
    assert len(new_expr.expressions) == 1
    assert And(Not(LessThan(Symbol("x"), Symbol("y")))) in new_expr.expressions


def test_Not_LessThan_concrete_scalar():
    transformer = DnfTransformer()

    ctx = get_context()
    x, y = Symbol("x"), Symbol("y")
    ctx.shapes[x] = ()
    ctx.types[x] = np.float32
    ctx.shapes[y] = ()
    ctx.types[y] = np.float32
    expr = Not(LessThan(x, y))
    new_expr = transformer.visit(expr)
    assert new_expr is not expr
    assert isinstance(new_expr, Or)
    assert len(new_expr.expressions) == 1
    assert And(GreaterThanOrEqual(x, y)) in new_expr.expressions


def test_Not_LessThan_concrete_scalar_array():
    transformer = DnfTransformer()

    ctx = get_context()
    x, y = Symbol("x"), Symbol("y")
    ctx.shapes[x] = (1, 1, 1, 1)
    ctx.types[x] = np.float32
    ctx.shapes[y] = (1, 1, 1, 1)
    ctx.types[y] = np.float32
    expr = Not(LessThan(x, y))
    new_expr = transformer.visit(expr)
    assert new_expr is not expr
    assert isinstance(new_expr, Or)
    assert len(new_expr.expressions) == 1
    assert And(GreaterThanOrEqual(x, y)) in new_expr.expressions


def test_Not_LessThan_concrete_shapes():
    transformer = DnfTransformer()

    ctx = get_context()
    x, y = Symbol("x"), Symbol("y")
    ctx.shapes[x] = (1, 3)
    ctx.types[x] = np.float32
    ctx.shapes[y] = (1, 3)
    ctx.types[y] = np.float32
    expr = Not(LessThan(x, y))
    new_expr = transformer.visit(expr)
    assert new_expr is not expr
    assert isinstance(new_expr, Or)
    assert len(new_expr.expressions) == 3
    assert And(GreaterThanOrEqual(x[0, 0], y[0, 0])) in new_expr.expressions
    assert And(GreaterThanOrEqual(x[0, 1], y[0, 1])) in new_expr.expressions
    assert And(GreaterThanOrEqual(x[0, 2], y[0, 2])) in new_expr.expressions


def test_Not_LessThan_concrete_shapes_broadcast_x():
    transformer = DnfTransformer()

    ctx = get_context()
    x, y = Constant(11), Symbol("y")
    ctx.shapes[y] = (1, 1, 1, 3)
    ctx.types[y] = np.float32
    expr = Not(LessThan(x, y))
    new_expr = transformer.visit(expr)
    assert new_expr is not expr
    assert isinstance(new_expr, Or)
    assert len(new_expr.expressions) == 3
    assert And(GreaterThanOrEqual(x, y[0, 0, 0, 0])) in new_expr.expressions
    assert And(GreaterThanOrEqual(x, y[0, 0, 0, 1])) in new_expr.expressions
    assert And(GreaterThanOrEqual(x, y[0, 0, 0, 2])) in new_expr.expressions


def test_Not_LessThan_concrete_shapes_broadcast_y():
    transformer = DnfTransformer()

    ctx = get_context()
    x, y = Symbol("x"), Constant(1)
    ctx.shapes[x] = (1, 3)
    ctx.types[x] = np.float32
    expr = Not(LessThan(x, y))
    new_expr = transformer.visit(expr)
    assert new_expr is not expr
    assert isinstance(new_expr, Or)
    assert len(new_expr.expressions) == 3
    assert And(GreaterThanOrEqual(x[0, 0], y)) in new_expr.expressions
    assert And(GreaterThanOrEqual(x[0, 1], y)) in new_expr.expressions
    assert And(GreaterThanOrEqual(x[0, 2], y)) in new_expr.expressions
