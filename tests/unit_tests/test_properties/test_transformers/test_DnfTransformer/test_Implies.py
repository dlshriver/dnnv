from dnnv.properties import *
from dnnv.properties.transformers import DnfTransformer


def test_Implies():
    transformer = DnfTransformer()

    expr = Implies(Symbol("x"), Symbol("y"))
    new_expr = transformer.visit(expr)
    assert new_expr is not expr
    assert isinstance(new_expr, Or)
    assert len(new_expr.expressions) == 2
    assert And(Not(Symbol("x"))) in new_expr.expressions
    assert And(Symbol("y")) in new_expr.expressions
