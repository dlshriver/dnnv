from dnnv.properties.expressions import *
from dnnv.properties.transformers import PropagateConstants


def test_Network():
    transformer = PropagateConstants()

    expr = Network("N")
    new_expr = transformer.visit(expr)
    assert new_expr is expr
