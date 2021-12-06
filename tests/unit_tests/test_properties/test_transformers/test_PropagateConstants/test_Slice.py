from dnnv.properties.expressions import *
from dnnv.properties.transformers import PropagateConstants


def test_Slice_non_concrete():
    transformer = PropagateConstants()

    expr = Slice(Symbol("s"), Symbol("e"), Constant(None))
    new_expr = transformer.visit(expr)
    assert new_expr is not expr
    assert isinstance(new_expr, Slice)
    assert new_expr.expr1 is expr.expr1
    assert new_expr.expr2 is expr.expr2
    assert new_expr.expr3 is expr.expr3


def test_Slice_concrete():
    transformer = PropagateConstants()

    expr = Slice(Constant(1), Constant(5), Constant(None))
    new_expr = transformer.visit(expr)
    assert new_expr is not expr
    assert isinstance(new_expr, Constant)
    assert new_expr.value == slice(1, 5, None)
