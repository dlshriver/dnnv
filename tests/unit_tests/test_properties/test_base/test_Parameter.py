import pytest

from dnnv.properties.base import *


def test_build_identifier():
    identifier = Parameter.build_identifier("eps", float)
    assert identifier == "eps"

    identifier = Parameter.build_identifier(Constant("param"), int)
    assert identifier == "param"

    with pytest.raises(TypeError):
        Parameter.build_identifier(31, int)


def test_repr():
    x = Parameter("param1", int)
    assert repr(x) == "Parameter('param1', type=int, default=None)"
    x.concretize(8)
    assert repr(x) == "8"

    x = Parameter("param2", float, 3.14)
    assert repr(x) == "Parameter('param2', type=float, default=3.14)"

    x = Parameter("param3", str, Constant("test"))
    assert repr(x) == "Parameter('param3', type=str, default='test')"


def test_str():
    x = Parameter("param1", int)
    assert str(x) == "Parameter('param1', type=int, default=None)"
    x.concretize(8)
    assert str(x) == "8"

    x = Parameter("param2", float, 3.14)
    assert str(x) == "Parameter('param2', type=float, default=3.14)"

    x = Parameter("param3", str, Constant("test"))
    assert str(x) == "Parameter('param3', type=str, default='test')"
