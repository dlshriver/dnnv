from dnnv.properties import *
from dnnv.properties.transformers import CnfTransformer


def test_Implies():
    transformer = CnfTransformer()

    expr = Implies(Symbol("x"), Symbol("y"))
    new_expr = transformer.visit(expr)
    assert new_expr is not expr
    assert isinstance(new_expr, And)
    assert len(new_expr.expressions) == 1
    new_expr_ = list(new_expr.expressions)[0]
    assert isinstance(new_expr_, Or)
    assert len(new_expr_.expressions) == 2
    assert Not(Symbol("x")) in new_expr_.expressions
    assert Symbol("y") in new_expr_.expressions
