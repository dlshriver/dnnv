from dnnv.properties.expressions import *
from dnnv.properties.transformers import PropagateConstants


def test_Parameter_non_concrete():
    transformer = PropagateConstants()

    expr = Parameter("param", int, 1)
    new_expr = transformer.visit(expr)
    assert new_expr is expr


def test_Parameter_concrete():
    transformer = PropagateConstants()

    expr = Parameter("param", int, 1)
    expr.concretize(param=5)
    new_expr = transformer.visit(expr)
    assert new_expr is not expr
    assert isinstance(new_expr, Constant)
    assert new_expr.value == 5
