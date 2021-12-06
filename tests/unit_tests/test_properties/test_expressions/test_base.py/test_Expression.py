import pytest

from dnnv.properties.expressions import *


def test___new__():
    with pytest.raises(TypeError, match="Expression may not be instantiated"):
        _ = Expression()


def test_Expression___getattr__():
    expr = Symbol("a")

    a1 = expr.a1
    assert isinstance(a1, Attribute)
    assert a1.expr1 is expr
    assert a1.expr2 is Constant("a1")

    a2 = getattr(expr, "a2")
    assert isinstance(a2, Attribute)
    assert a2.expr1 is expr
    assert a2.expr2 is Constant("a2")

    a3 = expr.__getattr__(Constant("a3"))
    assert isinstance(a3, Attribute)
    assert a3.expr1 is expr
    assert a3.expr2 is Constant("a3")

    a4 = expr.__getattr__(Symbol("a4"))
    assert isinstance(a4, Attribute)
    assert a4.expr1 is expr
    assert a4.expr2 is Symbol("a4")

    expr.concretize("string!")
    a5 = expr.__getattr__(Constant("strip"))
    assert isinstance(a5, Constant)
    assert a5 is Constant("string!".strip)

    a6 = expr.strip
    assert isinstance(a6, Constant)
    assert a6 is Constant("string!".strip)


def test_Expression_concretize():
    a = Constant("a")
    with pytest.raises(
        ValueError, match="Cannot concretize expression of type 'Constant'"
    ):
        a.concretize(1)

    a = Symbol("a")
    with pytest.raises(
        ValueError,
        match="Cannot specify both keyword and positional arguments for method 'concretize'",
    ):
        a.concretize(1, a=2)
    with pytest.raises(
        ValueError, match="Method 'concretize' expects at most 1 positional argument"
    ):
        a.concretize(1, 2)
    with pytest.raises(
        ValueError, match="Method 'concretize' expects at least 1 argument"
    ):
        a.concretize()
    assert not a.is_concrete

    a.concretize("a")
    assert a.is_concrete
    assert a.value == "a"

    b = Symbol("b")
    b.concretize(b="b")
    assert b.is_concrete
    assert b.value == "b"

    c = Symbol("c")
    with pytest.raises(
        ValueError,
        match="Unknown identifier 'd' for method 'concretize",
    ):
        c.concretize(d="d")


def test_Expression_value():
    class MockExpression(Expression):
        pass

    with pytest.raises(ValueError, match="Cannot get value of non-concrete expression"):
        _ = MockExpression().value
