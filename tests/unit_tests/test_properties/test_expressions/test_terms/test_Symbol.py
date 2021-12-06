import pytest

from dnnv.properties.expressions import *


def test_caching():
    symbol_a = Symbol("x")
    symbol_b = Symbol("x")
    assert symbol_a is symbol_b


def test_build_identifier():
    identifier = Symbol.build_identifier("a")
    assert identifier == "a"

    identifier = Symbol.build_identifier(Constant("b"))
    assert identifier == "b"

    with pytest.raises(TypeError):
        Symbol.build_identifier(31)


def test_value():
    x = Symbol("a")
    with pytest.raises(ValueError) as excinfo:
        _ = x.value
    assert str(excinfo.value) == "Cannot get value of non-concrete symbol."

    x = Symbol("b")
    x.concretize(513)
    x_ = x.value
    assert x_ == 513


def test_is_concrete():
    x = Symbol("a")
    assert not x.is_concrete

    x = Symbol("b")
    x.concretize(5)
    assert x.is_concrete


def test_concretize():
    x = Symbol("x")
    assert x._value is None
    x.concretize("apples")
    assert x._value == "apples"


def test_repr():
    x = Symbol("a")
    assert repr(x) == "Symbol('a')"


def test_str():
    x = Symbol("x")
    assert str(x) == "x"
