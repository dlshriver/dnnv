import numpy as np

from dnnv.properties.expressions import *
from dnnv.properties.transformers import SubstituteCalls


def test_numpy_shape():
    x = Symbol("x")
    x.ctx.shapes[x] = (2, 3)
    x.ctx.types[x] = np.float32
    expr = Constant(np.shape)(x)
    new_expr = SubstituteCalls().visit(expr)
    assert new_expr is not expr
    assert new_expr is Constant((2, 3))


def test_len():
    x = Symbol("x")
    x.ctx.shapes[x] = (2, 3)
    x.ctx.types[x] = np.float32
    expr = Constant(len)(x)
    new_expr = SubstituteCalls().visit(expr)
    assert new_expr is not expr
    assert new_expr is Constant(2)


def test_numpy_shape_no_shape():
    x = Symbol("x")
    expr = Constant(np.shape)(x)
    new_expr = SubstituteCalls().visit(expr)
    assert new_expr.is_equivalent(expr)


def test_len_no_shape():
    x = Symbol("x")
    expr = Constant(len)(x)
    new_expr = SubstituteCalls().visit(expr)
    assert new_expr.is_equivalent(expr)
