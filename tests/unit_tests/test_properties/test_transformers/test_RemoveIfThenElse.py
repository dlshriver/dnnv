import pytest

from dnnv.properties.expressions import *
from dnnv.properties.transformers import RemoveIfThenElse


def test_missing():
    class MockExpression(Expression):
        pass

    expr = MockExpression()
    new_expr = RemoveIfThenElse().visit(expr)
    assert new_expr is expr
    del MockExpression


def test_IfThenElse():
    transformer = RemoveIfThenElse()

    expr = IfThenElse(Symbol("condition"), Symbol("T"), Symbol("F"))
    new_expr = transformer.visit(expr)
    assert new_expr is not expr
    assert isinstance(new_expr, And)
    assert len(new_expr.expressions) == 2
    for e in new_expr.expressions:
        assert str(e) in ("(condition ==> T)", "(~condition ==> F)")
