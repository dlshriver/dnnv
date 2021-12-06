from dnnv.properties.expressions import *
from dnnv.properties.transformers import PropagateConstants


def test_Symbol_non_concrete():
    transformer = PropagateConstants()

    expr = Symbol("x")
    new_expr = transformer.visit(expr)
    assert new_expr is expr


def test_Symbol_concrete():
    transformer = PropagateConstants()

    expr = Symbol("x")
    expr.concretize(x=5)
    new_expr = transformer.visit(expr)
    assert new_expr is not expr
    assert isinstance(new_expr, Constant)
    assert new_expr.value == 5
