import pytest

from dnnv.properties.base import ArithmeticExpression


def test_call():
    expr = ArithmeticExpression()
    with pytest.raises(ValueError) as excinfo:
        _ = expr()
    assert str(excinfo.value) == "Arithmetic expressions are not callable."
