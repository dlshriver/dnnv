import pytest
import re

from dnnv.properties.expressions import *


def test_Multiply_symbols():
    a = Symbol("a")
    b = Symbol("b")

    c_1 = a * b
    c_2 = Multiply(a, b)

    assert c_1.is_equivalent(c_2)
    assert isinstance(c_1, ArithmeticExpression)


def test_Multiply_constants():
    a = Constant(1)
    b = Constant(2)

    c_1 = a * b
    c_2 = Multiply(a, b)
    assert c_1.is_equivalent(c_2)
    assert isinstance(c_1, ArithmeticExpression)
    assert c_1.is_concrete
    assert c_1.value == 2


def test_Multiply_mixed():
    a = Symbol("x")
    b = Constant(1)

    c_1 = a * b
    c_2 = Multiply(a, b)

    assert c_1.is_equivalent(c_2)
    assert isinstance(c_1, ArithmeticExpression)


def test_Multiply_non_arithmetic():
    class MockExpression(Expression):
        pass

    a = Symbol("a")
    b = MockExpression()

    with pytest.raises(
        TypeError,
        match=re.escape(
            "unsupported operand type(s) for *: 'Symbol' and 'MockExpression'"
        ),
    ):
        _ = a * b


def test_Multiply_non_expression():
    a = Symbol("x")
    b = Constant(1)

    c_1 = a * 1
    c_2 = Multiply(a, b)

    assert c_1.is_equivalent(c_2)
    assert isinstance(c_1, ArithmeticExpression)


def test_rMultiply():
    a = Symbol("x")
    b = Constant(1)

    c_1 = 1 * a
    c_2 = Multiply(b, a)

    assert c_1.is_equivalent(c_2)
    assert isinstance(c_1, ArithmeticExpression)


def test_rMultiply_non_arithmetic():
    class MockExpression(Expression):
        pass

    a = Symbol("a")
    b = MockExpression()

    with pytest.raises(
        TypeError,
        match=re.escape(
            "unsupported operand type(s) for *: 'MockExpression' and 'Symbol'"
        ),
    ):
        _ = b * a
