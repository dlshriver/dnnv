import numpy as np
import pytest

from dnnv.properties.base import Constant, Network, Symbol, _BUILTIN_FUNCTIONS


def test_abs():
    assert _BUILTIN_FUNCTIONS[abs] is _BUILTIN_FUNCTIONS[np.abs]

    abs_func = _BUILTIN_FUNCTIONS[abs]

    assert abs_func(5) is Constant(5)
    assert abs_func(-13) is Constant(13)

    x = Symbol("x")
    x.concretize(49)
    assert abs_func(x) is Constant(49)
    y = Symbol("y")
    y.concretize(91)
    assert abs_func(y) is Constant(91)

    a = Symbol("a")
    abs_a = abs_func(a)
    assert (
        repr(abs_a)
        == "IfThenElse(GreaterThanOrEqual(Symbol('a'), 0), Symbol('a'), Negation(Symbol('a')))"
    )


def test_argmax():
    argmax = _BUILTIN_FUNCTIONS[np.argmax]

    x = np.random.randn(1, 10)
    argmax_x = np.argmax(x)
    assert argmax(x) is Constant(argmax_x)

    y = Symbol("y")
    y_ = np.random.randn(1, 32)
    argmax_y = np.argmax(y_)
    y.concretize(y_)
    assert argmax(y) is Constant(argmax_y)

    with pytest.raises(RuntimeError, match=r"Unsupported type for argcmp input: (.*)"):
        _ = argmax(Symbol("a"))

    N = Network("N")
    x = Symbol("x")
    with pytest.raises(
        RuntimeError,
        match="argcmp can not be applied to outputs of non-concrete networks",
    ):
        _ = argmax(N(x))
    fake_network = lambda x: x
    fake_network.output_shape = ((1, 3),)
    N.concretize(fake_network)
    argmax_N_x = argmax(N(x))
    assert (
        repr(argmax_N_x)
        == "IfThenElse(And(GreaterThanOrEqual(Network('N')(Symbol('x'))[(0, 0)], Network('N')(Symbol('x'))[(0, 1)]), GreaterThanOrEqual(Network('N')(Symbol('x'))[(0, 0)], Network('N')(Symbol('x'))[(0, 2)])), 0, IfThenElse(And(GreaterThanOrEqual(Network('N')(Symbol('x'))[(0, 1)], Network('N')(Symbol('x'))[(0, 2)])), 1, 2))"
    )

    assert argmax_N_x.concretize(x=np.array([[0, 0, 0]])).value == 0
    assert argmax_N_x.concretize(x=np.array([[1, 0, 0]])).value == 0
    assert argmax_N_x.concretize(x=np.array([[2, 1, 0]])).value == 0
    assert argmax_N_x.concretize(x=np.array([[2, 0, 1]])).value == 0
    assert argmax_N_x.concretize(x=np.array([[3, 1, 2]])).value == 0
    assert argmax_N_x.concretize(x=np.array([[3, 2, 1]])).value == 0
    assert argmax_N_x.concretize(x=np.array([[0, 1, 0]])).value == 1
    assert argmax_N_x.concretize(x=np.array([[0, 1, 1]])).value == 1
    assert argmax_N_x.concretize(x=np.array([[1, 2, 0]])).value == 1
    assert argmax_N_x.concretize(x=np.array([[0, 0, 1]])).value == 2
    assert argmax_N_x.concretize(x=np.array([[1, 0, 2]])).value == 2
