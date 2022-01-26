from dnnv.properties import *
from dnnv.properties.transformers import DnfTransformer


def test_Symbol():
    transformer = DnfTransformer()

    expr = Symbol("x")
    new_expr = transformer.visit(expr)
    assert new_expr is not expr
    assert isinstance(new_expr, Or)
    assert And(Symbol("x")) in new_expr.expressions
