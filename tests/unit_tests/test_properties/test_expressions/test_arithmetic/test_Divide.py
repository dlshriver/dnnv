import pytest
import re

from dnnv.properties.expressions import *


def test_Divide_symbols():
    a = Symbol("a")
    b = Symbol("b")

    c_1 = a / b
    c_2 = Divide(a, b)

    assert c_1.is_equivalent(c_2)
    assert isinstance(c_1, ArithmeticExpression)


def test_Divide_constants():
    a = Constant(1)
    b = Constant(2)

    c_1 = a / b
    c_2 = Divide(a, b)
    assert c_1.is_equivalent(c_2)
    assert isinstance(c_1, ArithmeticExpression)
    assert c_1.is_concrete
    assert c_1.value == 0.5


def test_Divide_mixed():
    a = Symbol("x")
    b = Constant("1")

    c_1 = a / b
    c_2 = Divide(a, b)

    assert c_1.is_equivalent(c_2)
    assert isinstance(c_1, ArithmeticExpression)

    c_3 = b / a
    c_4 = Divide(b, a)

    assert c_3.is_equivalent(c_4)
    assert isinstance(c_3, ArithmeticExpression)


def test_Divide_non_arithmetic():
    class MockExpression(Expression):
        pass

    a = Symbol("a")
    b = MockExpression()

    with pytest.raises(
        TypeError,
        match=re.escape(
            "unsupported operand type(s) for /: 'Symbol' and 'MockExpression'"
        ),
    ):
        _ = a / b


def test_Divide_non_expression():
    a = Symbol("x")
    b = Constant(1)

    c_1 = a / 1
    c_2 = Divide(a, b)

    assert c_1.is_equivalent(c_2)
    assert isinstance(c_1, ArithmeticExpression)


def test_rDivide():
    a = Symbol("x")
    b = Constant(1)

    c_1 = 1 / a
    c_2 = Divide(b, a)

    assert c_1.is_equivalent(c_2)
    assert isinstance(c_1, ArithmeticExpression)


def test_rDivide_non_arithmetic():
    class MockExpression(Expression):
        pass

    a = Symbol("a")
    b = MockExpression()

    with pytest.raises(
        TypeError,
        match=re.escape(
            "unsupported operand type(s) for /: 'MockExpression' and 'Symbol'"
        ),
    ):
        _ = b / a
