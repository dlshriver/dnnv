import pytest

from dnnv.properties.expressions import *
from dnnv.properties.transformers import RemoveIfThenElse


def test_missing():
    class FakeExpression(Expression):
        pass

    with pytest.raises(
        ValueError, match="Unimplemented expression type: FakeExpression"
    ):
        RemoveIfThenElse().generic_visit(FakeExpression())
    del FakeExpression


def test_IfThenElse_cnf():
    transformer = RemoveIfThenElse(form="cnf")

    expr = IfThenElse(Symbol("condition"), Symbol("T"), Symbol("F"))
    new_expr = transformer.visit(expr)
    assert new_expr is not expr
    assert isinstance(new_expr, And)
    assert len(new_expr.expressions) == 2
    for e in new_expr.expressions:
        assert str(e) in ("(condition ==> T)", "(~condition ==> F)")


def test_IfThenElse_cnf_not():
    transformer = RemoveIfThenElse(form="cnf")

    expr = Not(IfThenElse(Symbol("condition"), Symbol("T"), Symbol("F")))
    new_expr = transformer.visit(expr)
    assert new_expr is not expr
    assert isinstance(new_expr, Not)
    assert isinstance(new_expr.expr, Or)
    assert len(new_expr.expr.expressions) == 2
    for e in new_expr.expr.expressions:
        assert str(e) in ("(condition & T)", "(~condition & F)")


def test_IfThenElse_dnf():
    transformer = RemoveIfThenElse(form="dnf")

    expr = IfThenElse(Symbol("condition"), Symbol("T"), Symbol("F"))
    new_expr = transformer.visit(expr)
    assert new_expr is not expr
    assert isinstance(new_expr, Or)
    assert len(new_expr.expressions) == 2
    for e in new_expr.expressions:
        assert str(e) in ("(condition & T)", "(~condition & F)")


def test_IfThenElse_dnf_not():
    transformer = RemoveIfThenElse(form="dnf")

    expr = Not(IfThenElse(Symbol("condition"), Symbol("T"), Symbol("F")))
    new_expr = transformer.visit(expr)
    assert new_expr is not expr
    assert isinstance(new_expr, Not)
    assert isinstance(new_expr.expr, And)
    assert len(new_expr.expr.expressions) == 2
    for e in new_expr.expr.expressions:
        assert str(e) in ("(condition ==> T)", "(~condition ==> F)")
