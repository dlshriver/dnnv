import numpy as np

from dnnv.properties.expressions import *
from dnnv.properties.transformers import SubstituteCalls


def test_numpy_mean_constant():
    a = np.random.randn(3, 4, 5)
    expr = Constant(np.mean)(Constant(a))
    new_expr = SubstituteCalls().visit(expr)
    assert new_expr is not expr
    assert new_expr is Constant(a.mean())


def test_numpy_mean_scalar():
    x = Symbol("x")
    x.ctx.shapes[x] = ()
    x.ctx.types[x] = np.float32
    expr = Constant(np.mean)(x)
    new_expr = SubstituteCalls().visit(expr)
    assert new_expr is not expr
    assert new_expr is x


def test_numpy_mean_array():
    x = Symbol("x")
    x.ctx.shapes[x] = (2, 2)
    x.ctx.types[x] = np.float32
    expr = Constant(np.mean)(x)
    new_expr = SubstituteCalls().visit(expr)
    assert new_expr is not expr
    assert new_expr.is_equivalent(Add(x[0, 0], x[0, 1], x[1, 0], x[1, 1]) / Constant(4))


def test_numpy_mean_noshape():
    x = Symbol("x")
    expr = Constant(np.mean)(x)
    new_expr = SubstituteCalls().visit(expr)
    assert new_expr is not expr
    assert new_expr.is_equivalent(expr)
