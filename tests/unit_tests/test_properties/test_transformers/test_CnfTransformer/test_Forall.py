from dnnv.properties import *
from dnnv.properties.transformers import CnfTransformer


def test_Forall():
    transformer = CnfTransformer()

    expr = Forall(Symbol("x"), Symbol("x"))
    new_expr = transformer.visit(expr)
    assert new_expr is not expr
    assert isinstance(new_expr, And)
    assert Or(Symbol("x")) in new_expr.expressions
