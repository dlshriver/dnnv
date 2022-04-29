import numpy as np

from dnnv.properties.expressions import *
from dnnv.properties.transformers import SubstituteCalls


def test_numpy_sum_constant():
    a = np.random.randn(3, 4, 5)
    expr = Constant(np.sum)(Constant(a))
    new_expr = SubstituteCalls().visit(expr)
    assert new_expr is not expr
    assert new_expr is Constant(a.sum())


def test_numpy_sum_constant_with_initial():
    a = np.random.rand(3, 4, 5)
    expr = Constant(np.sum)(Constant(a), initial=Constant(2.0))
    new_expr = SubstituteCalls().visit(expr)
    assert new_expr is not expr
    assert new_expr is Constant(np.sum(a, initial=2.0))

    expr = Constant(np.sum)(Constant(a), initial=Constant(0.9))
    new_expr = SubstituteCalls().visit(expr)
    assert new_expr is not expr
    assert new_expr is Constant(np.sum(a, initial=0.9))


def test_numpy_sum_scalar():
    x = Symbol("x")
    x.ctx.shapes[x] = ()
    x.ctx.types[x] = np.float32
    expr = Constant(np.sum)(x)
    new_expr = SubstituteCalls().visit(expr)
    assert new_expr is not expr
    assert new_expr is x

    expr = Constant(np.sum)(x, initial=Constant(2.0))
    new_expr = SubstituteCalls().visit(expr)
    assert new_expr is not expr
    assert new_expr.is_equivalent(Add(x, Constant(2.0)))


def test_numpy_sum_array():
    x = Symbol("x")
    x.ctx.shapes[x] = (2, 2)
    x.ctx.types[x] = np.float32
    expr = Constant(np.sum)(x)
    new_expr = SubstituteCalls().visit(expr)
    assert new_expr is not expr
    assert new_expr.is_equivalent(Add(x[0, 0], x[0, 1], x[1, 0], x[1, 1]))

    expr = Constant(np.sum)(x, initial=Constant(2.0))
    new_expr = SubstituteCalls().visit(expr)
    assert new_expr is not expr
    assert new_expr.is_equivalent(
        Add(x[0, 0], x[0, 1], x[1, 0], x[1, 1], Constant(2.0))
    )


def test_numpy_sum_noshape():
    x = Symbol("x")
    expr = Constant(np.sum)(x)
    new_expr = SubstituteCalls().visit(expr)
    assert new_expr is not expr
    assert new_expr.is_equivalent(expr)


def test_builtin_sum_constant():
    a = [1, 2, 3]
    expr = Constant(sum)(Constant(a))
    new_expr = SubstituteCalls().visit(expr)
    assert new_expr is not expr
    assert new_expr is Constant(6)


def test_builtin_sum_constant_with_start():
    a = [1, 2, 3]
    expr = Constant(sum)(Constant(a), start=Constant(2.0))
    new_expr = SubstituteCalls().visit(expr)
    assert new_expr is not expr
    assert new_expr is Constant(8.0)

    expr = Constant(sum)(Constant(a), start=Constant(-0.9))
    new_expr = SubstituteCalls().visit(expr)
    assert new_expr is not expr
    assert new_expr is Constant(5.1)

    expr = Constant(sum)(Constant(a), Constant(2.0))
    new_expr = SubstituteCalls().visit(expr)
    assert new_expr is not expr
    assert new_expr is Constant(8.0)

    expr = Constant(sum)(Constant(a), Constant(-0.9))
    new_expr = SubstituteCalls().visit(expr)
    assert new_expr is not expr
    assert new_expr is Constant(5.1)


def test_builtin_sum_array():
    x = Symbol("x")
    x.ctx.shapes[x] = (4,)
    x.ctx.types[x] = np.float32
    expr = Constant(sum)(x)
    new_expr = SubstituteCalls().visit(expr)
    assert new_expr is not expr
    assert new_expr.is_equivalent(Add(Constant(0), x[(0,)], x[(1,)], x[(2,)], x[(3,)]))

    expr = Constant(sum)(x, start=Constant(2.0))
    new_expr = SubstituteCalls().visit(expr)
    assert new_expr is not expr
    assert new_expr.is_equivalent(
        Add(x[(0,)], x[(1,)], x[(2,)], x[(3,)], Constant(2.0))
    )


def test_builtin_sum_noshape():
    x = Symbol("x")
    expr = Constant(sum)(x)
    new_expr = SubstituteCalls().visit(expr)
    assert new_expr is not expr
    assert new_expr.is_equivalent(expr)
