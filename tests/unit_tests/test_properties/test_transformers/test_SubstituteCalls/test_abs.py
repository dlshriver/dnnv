import numpy as np

from dnnv.properties.expressions import *
from dnnv.properties.transformers import SubstituteCalls


def test_abs():
    expr = Constant(abs)(Symbol("x"))
    new_expr = SubstituteCalls().visit(expr)
    assert new_expr is not expr
    assert new_expr.is_equivalent(
        IfThenElse(Symbol("x") >= Constant(0), Symbol("x"), -Symbol("x"))
    )


def test_numpy_abs():
    expr = Constant(np.abs)(Symbol("x"))
    new_expr = SubstituteCalls().visit(expr)
    assert new_expr is not expr
    assert new_expr.is_equivalent(
        IfThenElse(Symbol("x") >= Constant(0), Symbol("x"), -Symbol("x"))
    )


def test_abs_constant():
    expr = Call(Constant(abs), (10,), {})
    new_expr = SubstituteCalls().visit(expr)
    assert new_expr is not expr
    assert new_expr is Constant(10)
