import numpy as np

from dnnv.properties.expressions import *


def test_caching():
    const_a = Constant(238746)
    const_b = Constant(238746)
    assert const_a is const_b


def test_constant_constant():
    const_a = Constant(238746)
    const_b = Constant(const_a)
    assert const_a is const_b


def test_build_identifier():
    value = ["test", 5]
    value_type, identifier = Constant.build_identifier(value)
    assert value_type == list
    assert identifier == id(value)

    value_type, identifier = Constant.build_identifier(9)
    assert value_type == int
    assert identifier == 9

    value_type, identifier = Constant.build_identifier("test")
    assert value_type == str
    assert identifier == "test"


def test_is_concrete():
    const_true = Constant(True)
    assert const_true.is_concrete


def test_value():
    const = Constant(True)
    assert const.value == True

    const = Constant(9)
    assert const.value == 9

    const = Constant((1, 2, 3))
    assert const.value == (1, 2, 3)

    const = Constant(["a", "b", "c"])
    assert const.value == ["a", "b", "c"]

    const = Constant(set(["a", "b", "c"]))
    assert const.value == set(["a", "b", "c"])


def test_bool():
    const_true = Constant(True)
    assert const_true
    const_true = Constant(1)
    assert const_true
    const_true = Constant("test")
    assert const_true
    const_true = Constant([1, 2, 3])
    assert const_true

    const_false = Constant(False)
    assert not const_false
    const_false = Constant(0)
    assert not const_false
    const_false = Constant("")
    assert not const_false
    const_false = Constant(())
    assert not const_false


def test_repr():
    const = Constant(43)
    assert repr(const) == "43"
    const = Constant("string")
    assert repr(const) == "'string'"
    const = Constant((1, 2, 3))
    assert repr(const) == "(1, 2, 3)"

    const = Constant(slice(None, None, None))
    assert repr(const) == ":"
    const = Constant(slice(1, None, None))
    assert repr(const) == "1:"
    const = Constant(slice(None, 1, None))
    assert repr(const) == ":1"
    const = Constant(slice(None, None, 2))
    assert repr(const) == "::2"
    const = Constant(slice(0, -1, None))
    assert repr(const) == "0:-1"
    const = Constant(slice(0, -1, 2))
    assert repr(const) == "0:-1:2"

    const = Constant(np.array([1, 2, 3]))
    const_repr = repr(const)
    assert const_repr.startswith("np.ndarray{id=0x")
    assert const_repr.endswith(", shape=(3,), dtype=int64}")

    const = Constant(abs)
    assert repr(const) == "abs"


def test_str():
    const = Constant(43)
    assert str(const) == "43"
    const = Constant("string")
    assert str(const) == "'string'"
    const = Constant((1, 2, 3))
    assert str(const) == "(1, 2, 3)"

    const = Constant(slice(None, None, None))
    assert str(const) == ":"
    const = Constant(slice(1, None, None))
    assert str(const) == "1:"
    const = Constant(slice(None, 1, None))
    assert str(const) == ":1"
    const = Constant(slice(None, None, 2))
    assert str(const) == "::2"
    const = Constant(slice(0, -1, None))
    assert str(const) == "0:-1"
    const = Constant(slice(0, -1, 2))
    assert str(const) == "0:-1:2"

    const = Constant(np.array([1, 2, 3]))
    assert str(const) == "[1 2 3]"

    const = Constant(abs)
    assert str(const) == "abs"
