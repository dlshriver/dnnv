from dnnv.properties import *
from dnnv.properties.transformers import DnfTransformer


def test_Constant():
    transformer = DnfTransformer()

    expr = Constant(True)
    new_expr = transformer.visit(expr)
    assert new_expr is expr

    expr = Constant(False)
    new_expr = transformer.visit(expr)
    assert new_expr is expr
