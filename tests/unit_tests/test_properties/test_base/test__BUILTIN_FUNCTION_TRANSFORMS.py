import numpy as np
import pytest

from dnnv.properties.base import (
    Constant,
    Equal,
    FunctionCall,
    Network,
    NotEqual,
    Symbol,
    _BUILTIN_FUNCTION_TRANSFORMS,
)


def test_argcmp_eq():
    argcmp_eq = _BUILTIN_FUNCTION_TRANSFORMS[(np.argmax, Equal)]

    f_0_1 = Symbol("f")(Constant(0), Constant(1))
    with pytest.raises(RuntimeError, match="Too many arguments for argcmp"):
        argcmp_eq(f_0_1, Symbol("a"))
    f_kw = Symbol("f")(kwarg1=Constant(1))
    with pytest.raises(RuntimeError, match="Too many arguments for argcmp"):
        argcmp_eq(f_kw, Symbol("a"))
    f_a = Symbol("f")(Symbol("a"))
    assert argcmp_eq(f_a, Symbol("b")) is NotImplemented

    argmax = Symbol("np.argmax")
    argmax.concretize(np.argmax)
    N = Network("N")
    x = Symbol("x")
    with pytest.raises(
        RuntimeError,
        match="argcmp can not be applied to outputs of non-concrete networks",
    ):
        argcmp_eq(
            FunctionCall(argmax, (FunctionCall(N, (x,), {}),), {}),
            Symbol("C"),
        )
    fake_network = lambda x: x
    fake_network.output_shape = ((1, 3),)
    N.concretize(fake_network)

    a = np.random.randn(1, 3)
    argmax_a = np.argmax(a)
    assert argcmp_eq(
        FunctionCall(argmax, (FunctionCall(N, (Constant(a),), {}),), {}),
        Constant(argmax_a),
    ).value

    a = np.random.randn(1, 3)
    argmax_a = np.argmax(a)
    expr = argcmp_eq(
        FunctionCall(argmax, (FunctionCall(N, (Constant(a),), {}),), {}),
        Symbol("C"),
    )
    assert expr.concretize(C=argmax_a).value == True
    assert expr.concretize(C=argmax_a + 1).value == False
    assert expr.concretize(C=argmax_a - 1).value == False

    expr = argcmp_eq(
        FunctionCall(argmax, (FunctionCall(N, (x,), {}),), {}),
        Constant(1),
    )
    assert expr.concretize(x=np.array([[0, 0, 0]])).value == True
    assert expr.concretize(x=np.array([[1, 0, 0]])).value == False
    assert expr.concretize(x=np.array([[0, 1, 0]])).value == True
    assert expr.concretize(x=np.array([[0, 0, 1]])).value == False

    expr = argcmp_eq(
        FunctionCall(argmax, (FunctionCall(N, (Symbol("x0"),), {}),), {}),
        Constant(10),
    )
    assert expr.value == False


def test_argcmp_neq():
    argcmp_neq = _BUILTIN_FUNCTION_TRANSFORMS[(np.argmax, NotEqual)]

    f_0_1 = Symbol("f")(Constant(0), Constant(1))
    with pytest.raises(RuntimeError, match="Too many arguments for argcmp"):
        argcmp_neq(f_0_1, Symbol("a"))
    f_kw = Symbol("f")(kwarg1=Constant(1))
    with pytest.raises(RuntimeError, match="Too many arguments for argcmp"):
        argcmp_neq(f_kw, Symbol("a"))
    f_a = Symbol("f")(Symbol("a"))
    assert argcmp_neq(f_a, Symbol("b")) is NotImplemented

    argmax = Symbol("np.argmax")
    argmax.concretize(np.argmax)
    N = Network("N")
    x = Symbol("x")
    with pytest.raises(
        RuntimeError,
        match="argcmp can not be applied to outputs of non-concrete networks",
    ):
        argcmp_neq(
            FunctionCall(argmax, (FunctionCall(N, (x,), {}),), {}),
            Symbol("C"),
        )
    fake_network = lambda x: x
    fake_network.output_shape = ((1, 3),)
    N.concretize(fake_network)

    a = np.random.randn(1, 3)
    argmax_a = np.argmax(a)
    assert not argcmp_neq(
        FunctionCall(argmax, (FunctionCall(N, (Constant(a),), {}),), {}),
        Constant(argmax_a),
    ).value

    a = np.random.randn(1, 3)
    argmax_a = np.argmax(a)
    expr = argcmp_neq(
        FunctionCall(argmax, (FunctionCall(N, (Constant(a),), {}),), {}),
        Symbol("C"),
    )
    assert expr.concretize(C=argmax_a).value == False
    assert expr.concretize(C=argmax_a + 1).value == True
    assert expr.concretize(C=argmax_a - 1).value == True

    expr = argcmp_neq(
        FunctionCall(argmax, (FunctionCall(N, (x,), {}),), {}),
        Constant(1),
    )
    assert expr.concretize(x=np.array([[0, 0, 0]])).value == False
    assert expr.concretize(x=np.array([[1, 0, 0]])).value == True
    assert expr.concretize(x=np.array([[0, 1, 0]])).value == False
    assert expr.concretize(x=np.array([[0, 0, 1]])).value == True

    expr = argcmp_neq(
        FunctionCall(argmax, (FunctionCall(N, (Symbol("x0"),), {}),), {}),
        Constant(10),
    )
    assert expr.value == True
