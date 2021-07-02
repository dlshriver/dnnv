import pytest

from dnnv.properties.base import *


def test_Expression_new():
    with pytest.raises(TypeError) as excinfo:
        epxr = Expression()
    assert str(excinfo.value) == "Expression may not be instantiated"


def test_concretize_expression():
    a = Symbol("a")
    b = Symbol("b")

    expr = Add(a, b)
    assert not expr.is_concrete
    expr.concretize(a=3, b=2)
    assert expr.is_concrete
    assert expr.value == 5


def test_concretize_symbol():
    a = Symbol("a")

    expr = Add(a, Constant(5))
    assert not expr.is_concrete
    a.concretize(4)
    assert expr.is_concrete
    assert expr.value == 9

    b = Symbol("b")
    assert not b.is_concrete
    b.concretize(b=47)
    assert b.is_concrete
    assert b.value == 47


def test_concretize_positional_args_with_non_symbol():
    a = Symbol("a")
    b = Symbol("b")

    expr = Add(a, b)
    with pytest.raises(ValueError) as excinfo:
        expr.concretize(11, 13)
    assert str(excinfo.value).startswith("Cannot concretize expression of type")


def test_concretize_too_many_positional_args():
    a = Symbol("a")
    with pytest.raises(ValueError) as excinfo:
        a.concretize(11, 13)
    assert str(excinfo.value) == "'concretize' expects at most 1 positional argument"


def test_concretize_unknown_identifier():
    x = Symbol("x")
    with pytest.raises(ValueError) as excinfo:
        x.concretize(y=-5)
    assert str(excinfo.value) == "Unknown identifier: 'y'"


def test_networks():
    expr = And(Constant(False), Symbol("a"))
    assert len(expr.networks) == 0

    expr = Network("N")(Symbol("x")) < Constant(0)
    assert len(expr.networks) == 1

    expr = Network("N1")(Symbol("x")) >= Network("N2")(Symbol("x"))
    assert len(expr.networks) == 2


def test_parameters():
    expr = And(Constant(False), Symbol("a"))
    assert len(expr.parameters) == 0

    expr = Symbol("x")[Parameter("i", int, default=0)]
    assert len(expr.parameters) == 1
    expr = (
        Symbol("x")[Parameter("i", int, default=0)]
        < Symbol("x")[Parameter("i", int, default=0) + 1]
    )
    assert len(expr.parameters) == 1

    expr = (
        Symbol("x")[Parameter("i1", int, default=0)]
        < Symbol("x")[Parameter("i2", int, default=0)]
    )
    assert len(expr.parameters) == 2


def test_bool():
    expr = Symbol("F")
    expr.concretize(False)
    assert not bool(expr)

    expr = Symbol("T")
    expr.concretize(True)
    assert bool(expr)

    expr = And(Constant(False), Symbol("a"))
    with pytest.warns(
        RuntimeWarning,
        match="Using the bool representation of non-concrete expression can have unexpected results.",
    ):
        assert bool(expr)

    expr = Implies(Symbol("b"), Constant(False))
    with pytest.warns(
        RuntimeWarning,
        match="Using the bool representation of non-concrete expression can have unexpected results.",
    ):
        assert bool(expr)

    with pytest.warns(
        RuntimeWarning,
        match="Using the bool representation of non-concrete expression can have unexpected results.",
    ):
        assert bool(Symbol("z"))


def test_get_attr():
    x = Symbol("x")
    x.concretize("test")
    assert x.lower == "test".lower

    assert x.__getattr__(Constant("upper")) == "test".upper

    expr = x.__getattr__(Symbol("attr"))
    assert isinstance(expr, Attribute)
    assert expr.expr1 is x
    assert expr.expr2 is Symbol("attr")

    expr = Symbol("y").attribute
    assert isinstance(expr, Attribute)
    assert expr.expr1 is Symbol("y")
    assert expr.expr2 is Constant("attribute")


def test_get_item():
    x = Symbol("x")
    x.concretize("test")
    assert x[0] == "t"
    assert x[-2:] == "st"

    i = Symbol("i")
    x_i = x[i]
    assert isinstance(x_i, Subscript)
    assert x_i.expr1 is x
    assert x_i.expr2 is i
    i.concretize(1)
    assert x[i] == "e"


def test_add():
    a = Symbol("a")
    b = Symbol("b")

    expr = a + b
    assert isinstance(expr, Add)
    assert len(expr.expressions) == 2
    assert a in expr.expressions
    assert b in expr.expressions

    expr = a + 3
    assert isinstance(expr, Add)
    assert len(expr.expressions) == 2
    assert a in expr.expressions
    assert Constant(3) in expr.expressions

    expr = 3 + b
    assert isinstance(expr, Add)
    assert len(expr.expressions) == 2
    assert b in expr.expressions
    assert Constant(3) in expr.expressions


def test_sub():
    a = Symbol("a")
    b = Symbol("b")

    expr = a - b
    assert isinstance(expr, Subtract)
    assert expr.expr1 is a
    assert expr.expr2 is b

    expr = a - 3
    assert isinstance(expr, Subtract)
    assert expr.expr1 is a
    assert expr.expr2 is Constant(3)

    expr = 3 - b
    assert isinstance(expr, Subtract)
    assert expr.expr1 is Constant(3)
    assert expr.expr2 is b


def test_mul():
    a = Symbol("a")
    b = Symbol("b")

    expr = a * b
    assert isinstance(expr, Multiply)
    assert len(expr.expressions) == 2
    assert a in expr.expressions
    assert b in expr.expressions

    expr = a * 3
    assert isinstance(expr, Multiply)
    assert len(expr.expressions) == 2
    assert a in expr.expressions
    assert Constant(3) in expr.expressions

    expr = 3 * b
    assert isinstance(expr, Multiply)
    assert len(expr.expressions) == 2
    assert b in expr.expressions
    assert Constant(3) in expr.expressions


def test_truediv():
    a = Symbol("a")
    b = Symbol("b")

    expr = a / b
    assert isinstance(expr, Divide)
    assert expr.expr1 is a
    assert expr.expr2 is b

    expr = a / 3
    assert isinstance(expr, Divide)
    assert expr.expr1 is a
    assert expr.expr2 is Constant(3)

    expr = 3 / b
    assert isinstance(expr, Divide)
    assert expr.expr1 is Constant(3)
    assert expr.expr2 is b


def test_neg():
    a = Symbol("a")
    expr = -a
    assert isinstance(expr, Negation)
    assert expr.expr is a


def test_eq():
    a = Symbol("a")
    b = Symbol("b")

    expr = a == b
    assert isinstance(expr, Equal)
    assert expr.expr1 is a
    assert expr.expr2 is b

    expr = a == 3
    assert isinstance(expr, Equal)
    assert expr.expr1 is a
    assert expr.expr2 is Constant(3)

    expr = 3 == b
    assert isinstance(expr, Equal)
    assert expr.expr2 is Constant(3)
    assert expr.expr1 is b


def test_ne():
    a = Symbol("a")
    b = Symbol("b")

    expr = a != b
    assert isinstance(expr, NotEqual)
    assert expr.expr1 is a
    assert expr.expr2 is b

    expr = a != 3
    assert isinstance(expr, NotEqual)
    assert expr.expr1 is a
    assert expr.expr2 is Constant(3)

    expr = 3 != b
    assert isinstance(expr, NotEqual)
    assert expr.expr2 is Constant(3)
    assert expr.expr1 is b


def test_ge():
    a = Symbol("a")
    b = Symbol("b")

    expr = a >= b
    assert isinstance(expr, GreaterThanOrEqual)
    assert expr.expr1 is a
    assert expr.expr2 is b

    expr = a >= 3
    assert isinstance(expr, GreaterThanOrEqual)
    assert expr.expr1 is a
    assert expr.expr2 is Constant(3)

    expr = 3 >= b
    assert isinstance(expr, LessThanOrEqual)
    assert expr.expr2 is Constant(3)
    assert expr.expr1 is b


def test_gt():
    a = Symbol("a")
    b = Symbol("b")

    expr = a > b
    assert isinstance(expr, GreaterThan)
    assert expr.expr1 is a
    assert expr.expr2 is b

    expr = a > 3
    assert isinstance(expr, GreaterThan)
    assert expr.expr1 is a
    assert expr.expr2 is Constant(3)

    expr = 3 > b
    assert isinstance(expr, LessThan)
    assert expr.expr2 is Constant(3)
    assert expr.expr1 is b


def test_le():
    a = Symbol("a")
    b = Symbol("b")

    expr = a <= b
    assert isinstance(expr, LessThanOrEqual)
    assert expr.expr1 is a
    assert expr.expr2 is b

    expr = a <= 3
    assert isinstance(expr, LessThanOrEqual)
    assert expr.expr1 is a
    assert expr.expr2 is Constant(3)

    expr = 3 <= b
    assert isinstance(expr, GreaterThanOrEqual)
    assert expr.expr2 is Constant(3)
    assert expr.expr1 is b


def test_lt():
    a = Symbol("a")
    b = Symbol("b")

    expr = a < b
    assert isinstance(expr, LessThan)
    assert expr.expr1 is a
    assert expr.expr2 is b

    expr = a < 3
    assert isinstance(expr, LessThan)
    assert expr.expr1 is a
    assert expr.expr2 is Constant(3)

    expr = 3 < b
    assert isinstance(expr, GreaterThan)
    assert expr.expr2 is Constant(3)
    assert expr.expr1 is b


def test_and():
    a = Symbol("a")
    b = Symbol("b")

    expr = a & b
    assert isinstance(expr, And)
    assert len(expr.expressions) == 2
    assert a in expr.expressions
    assert b in expr.expressions

    expr = a & 3
    assert isinstance(expr, And)
    assert len(expr.expressions) == 2
    assert a in expr.expressions
    assert Constant(3) in expr.expressions

    expr = 3 & b
    assert isinstance(expr, And)
    assert len(expr.expressions) == 2
    assert Constant(3) in expr.expressions
    assert b in expr.expressions


def test_or():
    a = Symbol("a")
    b = Symbol("b")

    expr = a | b
    assert isinstance(expr, Or)
    assert len(expr.expressions) == 2
    assert a in expr.expressions
    assert b in expr.expressions

    expr = a | 3
    assert isinstance(expr, Or)
    assert len(expr.expressions) == 2
    assert a in expr.expressions
    assert Constant(3) in expr.expressions

    expr = 3 | b
    assert isinstance(expr, Or)
    assert len(expr.expressions) == 2
    assert Constant(3) in expr.expressions
    assert b in expr.expressions


def test_invert():
    a = Symbol("a")
    expr = ~a
    assert isinstance(expr, Not)
    assert expr.expr is a


def test_call():
    int_f = Symbol("int_f")
    int_f.concretize(int)
    x = int_f()
    assert isinstance(x, Constant)
    assert x is Constant(0)

    f = Symbol("f")
    y = f(Symbol("x"))
    assert isinstance(y, FunctionCall)
    assert y.function is f
    assert len(y.args) == 1
    assert y.args[0] is Symbol("x")
    assert len(y.kwargs) == 0
