from dnnv.properties.expressions import *
from dnnv.properties.transformers import PropagateConstants


def test_Constant():
    transformer = PropagateConstants()

    expr = Constant(1)
    new_expr = transformer.visit(expr)
    assert new_expr is expr
