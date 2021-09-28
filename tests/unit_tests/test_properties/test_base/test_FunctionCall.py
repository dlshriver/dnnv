import numpy as np
import pytest

import dnnv.nn.operations as operations

from dnnv.nn.graph import OperationGraph
from dnnv.properties.base import *


def test_value():
    func_call = FunctionCall(Constant(np.sign), (Constant(5),), {})
    assert func_call.value == 1

    func_call = FunctionCall(
        Constant(np.ones), (Constant((1, 5)),), {"dtype": Constant(np.float32)}
    )
    assert func_call.value.dtype == np.float32
    assert np.allclose(func_call.value, np.ones((1, 5), dtype=np.float32))

    func_call = FunctionCall(Constant(np.argmax), (Symbol("y"),), {})
    with pytest.raises(ValueError):
        _ = func_call.value

    input_op = operations.Input((-1, 5), np.dtype(np.float32))
    mul_op = operations.Mul(input_op, 2.0)
    add_op = operations.Add(mul_op, -1.0)
    relu_op = operations.Relu(add_op)
    op_graph = OperationGraph([relu_op])
    Network("N1").concretize(op_graph)
    input_op = operations.Input((-1, 5), np.dtype(np.float32))
    mul_op = operations.Mul(input_op, -2.0)
    add_op = operations.Add(mul_op, 10.0)
    relu_op = operations.Relu(add_op)
    op_graph = OperationGraph([relu_op])
    Network("N2").concretize(op_graph)
    func_call = FunctionCall(
        Constant(Network("N2").compose),
        (Network("N1"),),
        {},
    )
    N12 = func_call.value
    assert isinstance(N12, Network)
    assert repr(N12) == "Network('N2○N1')"


def test_repr():
    func_call = FunctionCall(Constant(np.sign), (Constant(5),), {})
    assert repr(func_call) == "numpy.sign(5)"

    func_call = FunctionCall(
        Constant(np.ones), (Constant((1, 5)),), {"dtype": Constant(np.float32)}
    )
    assert repr(func_call) == "numpy.ones((1, 5), dtype=numpy.float32)"

    func_call = FunctionCall(Network("N"), (Constant(5),), {})
    assert repr(func_call) == "Network('N')(5)"

    func_call = FunctionCall(Constant(dict), (), {})
    assert repr(func_call) == "builtins.dict()"

    func_call = FunctionCall(Constant(int), (Constant("11"),), {"base": Constant(2)})
    assert repr(func_call) == "builtins.int('11', base=2)"


def test_str():
    func_call = FunctionCall(Constant(np.sign), (Constant(5),), {})
    assert str(func_call) == "numpy.sign(5)"

    func_call = FunctionCall(
        Constant(np.ones), (Constant((1, 5)),), {"dtype": Constant(np.float32)}
    )
    assert str(func_call) == "numpy.ones((1, 5), dtype=numpy.float32)"

    func_call = FunctionCall(Network("N"), (Constant(5),), {})
    assert str(func_call) == "N(5)"

    func_call = FunctionCall(Constant(dict), (), {})
    assert str(func_call) == "builtins.dict()"

    func_call = FunctionCall(Constant(int), (Constant("11"),), {"base": Constant(2)})
    assert str(func_call) == "builtins.int(11, base=2)"
