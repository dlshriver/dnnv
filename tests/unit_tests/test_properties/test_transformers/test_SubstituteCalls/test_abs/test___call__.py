import numpy as np
import pytest

from dnnv.errors import DNNVError
from dnnv.properties.expressions import *
from dnnv.properties.transformers import SubstituteCalls


def test_abs():
    x = Symbol("x")
    x.ctx.shapes[x] = ()
    x.ctx.types[x] = np.float32
    expr = Constant(abs)(x)
    new_expr = SubstituteCalls().visit(expr)
    assert new_expr is not expr
    assert new_expr.is_equivalent(IfThenElse(x >= Constant(0), x, -x))


def test_abs_no_shape():
    x = Symbol("x")
    expr = Constant(abs)(x)
    with pytest.raises(DNNVError, match="Unsupported shape for expression"):
        _ = SubstituteCalls().visit(expr)


def test_abs_non_scalar():
    x = Symbol("x")
    x.ctx.shapes[x] = (3, 4)
    x.ctx.types[x] = np.float32
    expr = Constant(abs)(x)
    with pytest.raises(DNNVError, match="Unsupported shape for expression"):
        _ = SubstituteCalls().visit(expr)


def test_numpy_abs():
    x = Symbol("x")
    x.ctx.shapes[x] = ()
    x.ctx.types[x] = np.float32
    expr = Constant(np.abs)(x)
    new_expr = SubstituteCalls().visit(expr)
    assert new_expr is not expr
    assert new_expr.is_equivalent(IfThenElse(x >= Constant(0), x, -x))


def test_abs_constant():
    expr = Call(Constant(abs), (Constant(10),), {})
    new_expr = SubstituteCalls().visit(expr)
    assert new_expr is not expr
    assert new_expr is Constant(10)
