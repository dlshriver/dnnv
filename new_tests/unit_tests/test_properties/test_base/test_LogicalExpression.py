import pytest

from dnnv.properties.base import LogicalExpression, Not


def test_invert():
    expr = LogicalExpression()
    not_expr = ~expr
    assert isinstance(not_expr, Not)
    assert not_expr.expr is expr


def test_call():
    expr = LogicalExpression()
    with pytest.raises(ValueError) as excinfo:
        _ = expr()
    assert str(excinfo.value) == "Logical expressions are not callable."
