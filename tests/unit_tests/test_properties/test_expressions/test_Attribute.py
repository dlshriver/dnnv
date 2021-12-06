import pytest

from dnnv.properties.expressions import *


def test_Attribute():
    expr = Attribute(Symbol("a"), Symbol("b"))
    assert expr.expr1 == Symbol("a")
    assert expr.expr2 == Symbol("b")


def test_repr():
    expr = Attribute(Symbol("a"), Symbol("b"))
    assert repr(expr) == "Symbol('a').Symbol('b')"


def test_str():
    expr = Attribute(Symbol("a"), Symbol("b"))
    assert str(expr) == "a.b"


def test_value():
    expr = Attribute(Constant("test"), Constant("lower"))
    assert expr.value == "test".lower

    expr = Attribute(Symbol("a"), Symbol("b"))
    with pytest.raises(ValueError, match="Cannot get value of non-concrete expression"):
        _ = expr.value

    expr = Attribute(Constant("test_string"), Constant("fake_attribute"))
    with pytest.raises(
        ValueError, match="'str' object has no attribute 'fake_attribute'"
    ):
        _ = expr.value


def test_expr():
    expr = Attribute(Symbol("a"), Symbol("b"))
    assert expr.expr is expr.expr1


def test_expr():
    expr = Attribute(Symbol("a"), Symbol("b"))
    assert expr.name is expr.expr2
