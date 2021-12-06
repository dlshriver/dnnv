import pytest

from dnnv.properties.expressions import LogicalExpression, Not


def test_invert():
    expr = LogicalExpression()
    not_expr = ~expr
    assert isinstance(not_expr, Not)
    assert not_expr.expr is expr


def test_call():
    expr = LogicalExpression()
    with pytest.raises(TypeError, match="'LogicalExpression' object is not callable"):
        _ = expr()
