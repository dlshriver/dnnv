import pytest
import re

from dnnv.properties.expressions import *


def test_invert():
    expr = And(Symbol("x"), Symbol("y"))
    not_expr = ~expr
    assert isinstance(not_expr, Or)
    for e in not_expr.expressions:
        assert isinstance(e, Not)
        assert e.expr in expr.expressions


def test_And_symbols():
    a = Symbol("a")
    b = Symbol("b")

    c_1 = a & b
    c_2 = And(a, b)

    assert c_1.is_equivalent(c_2)
    assert isinstance(c_1, LogicalExpression)


def test_And_constants():
    a = Constant(True)
    b = Constant(False)

    c_1 = a & b
    c_2 = And(a, b)
    assert c_1.is_equivalent(c_2)
    assert isinstance(c_1, LogicalExpression)
    assert c_1.is_concrete
    assert c_1.value == False


def test_And_mixed():
    a = Symbol("x")
    b = Constant(True)

    c_1 = a & b
    c_2 = And(a, b)

    assert c_1.is_equivalent(c_2)
    assert isinstance(c_1, LogicalExpression)


def test_And_non_logical():
    class MockExpression(Expression):
        pass

    a = Symbol("a")
    b = MockExpression()

    with pytest.raises(
        TypeError,
        match=re.escape(
            "unsupported operand type(s) for &: 'Symbol' and 'MockExpression'"
        ),
    ):
        _ = a & b


def test_And_non_expression():
    a = Symbol("x")
    b = Constant(True)

    c_1 = a & 1
    c_2 = And(a, b)

    assert c_1.is_equivalent(c_2)
    assert isinstance(c_1, LogicalExpression)


def test_rAnd():
    a = Symbol("x")
    b = Constant(True)

    c_1 = True & a
    c_2 = And(b, a)

    assert c_1.is_equivalent(c_2)
    assert isinstance(c_1, LogicalExpression)


def test_rAnd_non_logical():
    class MockExpression(Expression):
        pass

    a = Symbol("a")
    b = MockExpression()

    with pytest.raises(
        TypeError,
        match=re.escape(
            "unsupported operand type(s) for &: 'MockExpression' and 'Symbol'"
        ),
    ):
        _ = b & a
