from dnnv.properties.base import *
from dnnv.properties.transformers import ToCNF


def test_Forall():
    transformer = ToCNF()

    expr = Forall(Symbol("x"), Symbol("x"))
    new_expr = transformer.visit(expr)
    assert new_expr is not expr
    assert isinstance(new_expr, And)
    assert Or(Symbol("x")) in new_expr.expressions
