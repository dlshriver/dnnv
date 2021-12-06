import pytest

from dnnv.properties.expressions import *
from dnnv.properties.transformers import RemoveIfThenElse


def test_missing():
    class FakeExpression(Expression):
        pass

    with pytest.raises(
        ValueError, match="Unimplemented expression type: FakeExpression"
    ):
        RemoveIfThenElse().visit(FakeExpression())
    del FakeExpression


def test_IfThenElse():
    transformer = RemoveIfThenElse()

    expr = IfThenElse(Symbol("condition"), Symbol("T"), Symbol("F"))
    new_expr = transformer.visit(expr)
    assert new_expr is not expr
    assert isinstance(new_expr, And)
    assert len(new_expr.expressions) == 2
    for e in new_expr.expressions:
        assert str(e) in ("(condition ==> T)", "(~condition ==> F)")
