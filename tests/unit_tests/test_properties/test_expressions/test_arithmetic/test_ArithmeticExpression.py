import pytest

from dnnv.properties.expressions.arithmetic import ArithmeticExpression


def test_call():
    expr = ArithmeticExpression()
    with pytest.raises(
        TypeError, match="'ArithmeticExpression' object is not callable"
    ):
        _ = expr()
