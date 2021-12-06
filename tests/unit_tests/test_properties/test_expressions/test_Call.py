import numpy as np
import pytest

import dnnv.nn.operations as operations

from dnnv.nn.graph import OperationGraph
from dnnv.properties.expressions import *


def test_value():
    func_call = Call(Constant(np.sign), (Constant(5),), {})
    assert func_call.value == 1

    func_call = Call(
        Constant(np.ones), (Constant((1, 5)),), {"dtype": Constant(np.float32)}
    )
    assert func_call.value.dtype == np.float32
    assert np.allclose(func_call.value, np.ones((1, 5), dtype=np.float32))

    func_call = Call(Constant(np.argmax), (Symbol("y"),), {})
    with pytest.raises(ValueError, match="Cannot get value of non-concrete expression"):
        _ = func_call.value

    input_op = operations.Input((-1, 5), np.dtype(np.float32))
    mul_op = operations.Mul(input_op, 2.0)
    add_op = operations.Add(mul_op, -1.0)
    relu_op = operations.Relu(add_op)
    op_graph_1 = OperationGraph([relu_op])
    Network("N1").concretize(op_graph_1)
    input_op = operations.Input((-1, 5), np.dtype(np.float32))
    mul_op = operations.Mul(input_op, -2.0)
    add_op = operations.Add(mul_op, 10.0)
    relu_op = operations.Relu(add_op)
    op_graph_2 = OperationGraph([relu_op])
    Network("N2").concretize(op_graph_2)
    func_call = Call(
        Network("N2").compose,
        (Network("N1"),),
        {},
    )
    N12 = func_call.value
    assert isinstance(N12, OperationGraph)


def test_repr():
    func_call = Call(Constant(np.sign), (Constant(5),), {})
    assert repr(func_call) == "numpy.sign(5)"

    func_call = Call(
        Constant(np.ones), (Constant((1, 5)),), {"dtype": Constant(np.float32)}
    )
    assert repr(func_call) == "numpy.ones((1, 5), dtype=numpy.float32)"

    func_call = Call(Network("N"), (Constant(5),), {})
    assert repr(func_call) == "Network('N')(5)"

    func_call = Call(Constant(dict), (), {})
    assert repr(func_call) == "builtins.dict()"

    func_call = Call(Constant(int), (Constant("11"),), {"base": Constant(2)})
    assert repr(func_call) == "builtins.int('11', base=2)"

    func_call = Call(Constant(str), (), {"encoding": Constant("utf8")})
    assert repr(func_call) == "builtins.str(encoding='utf8')"


def test_str():
    func_call = Call(Constant(np.sign), (Constant(5),), {})
    assert str(func_call) == "numpy.sign(5)"

    func_call = Call(
        Constant(np.ones), (Constant((1, 5)),), {"dtype": Constant(np.float32)}
    )
    assert str(func_call) == "numpy.ones((1, 5), dtype=numpy.float32)"

    func_call = Call(Network("N"), (Constant(5),), {})
    assert str(func_call) == "N(5)"

    func_call = Call(Constant(dict), (), {})
    assert str(func_call) == "builtins.dict()"

    func_call = Call(Constant(int), (Constant("11"),), {"base": Constant(2)})
    assert str(func_call) == "builtins.int('11', base=2)"

    func_call = Call(Constant(str), (), {"encoding": Constant("utf8")})
    assert str(func_call) == "builtins.str(encoding='utf8')"


def test_is_equivalent():
    expr1 = Call(Constant(abs), (Constant(-4),), {})
    expr2 = Call(Constant(abs), (Constant(-4),), {})
    expr3 = Call(Constant(abs), (Constant(-2),), {})
    expr4 = Call(Constant(hex), (Constant(-4),), {})

    assert expr1.is_equivalent(expr1)
    assert expr1.is_equivalent(expr2)
    assert expr2.is_equivalent(expr1)
    assert not expr1.is_equivalent(expr3)
    assert not expr3.is_equivalent(expr1)
    assert not expr1.is_equivalent(expr4)
    assert not expr4.is_equivalent(expr1)
