import numpy as np
import pytest

import dnnv.nn.operations as operations

from dnnv.nn.graph import OperationGraph
from dnnv.properties.base import *


def test_new():
    N = Network()
    assert N.identifier == "N"
    assert str(N) == "N"
    N_ = Network()
    assert N_ is N

    N = Network("DNN")
    assert N.identifier == "DNN"
    assert str(N) == "DNN"


def test_repr():
    N = Network()
    assert repr(N) == "Network('N')"

    N = Network("DNN")
    assert repr(N) == "Network('DNN')"


def test_get_item():
    N = Network("N")
    N_0 = N[0]
    assert isinstance(N_0, Subscript)
    assert N_0.expr == N

    N_0 = N[:-1]
    assert isinstance(N_0, Subscript)
    assert N_0.expr == N

    input_op = operations.Input((-1, 5), np.dtype(np.float32))
    mul_op = operations.Mul(input_op, 2.0)
    add_op = operations.Add(mul_op, -1.0)
    relu_op = operations.Relu(add_op)
    op_graph = OperationGraph([relu_op])
    N.concretize(op_graph)

    N_0 = N[Constant(0)]
    assert isinstance(N_0, Network)
    assert repr(N_0) == "Network('N[0]')"

    N_0 = N[0]
    assert isinstance(N_0, Network)
    assert repr(N_0) == "Network('N[0]')"

    N_ = N[:-1]
    assert isinstance(N_, Network)
    assert repr(N_) == "Network('N[:-1]')"

    N_ = N[:-1:1]
    assert isinstance(N_, Network)
    assert repr(N_) == "Network('N[:-1:1]')"


def test_compose():
    N = Network("N")
    x = Symbol("x")
    with pytest.raises(ValueError) as excinfo:
        _ = N.compose(x)
    assert str(excinfo.value) == "Networks can only be composed with other networks."

    N_0 = Network("N_0")
    with pytest.raises(ValueError) as excinfo:
        _ = N.compose(N_0)
    assert str(excinfo.value) == "Network is not concrete."

    input_op = operations.Input((-1, 5), np.dtype(np.float32))
    mul_op = operations.Mul(input_op, 2.0)
    add_op = operations.Add(mul_op, -1.0)
    relu_op = operations.Relu(add_op)
    op_graph = OperationGraph([relu_op])
    N.concretize(op_graph)

    input_op = operations.Input((-1, 5), np.dtype(np.float32))
    mul_op = operations.Mul(input_op, -2.0)
    add_op = operations.Add(mul_op, 10.0)
    relu_op = operations.Relu(add_op)
    op_graph = OperationGraph([relu_op])
    N_0.concretize(op_graph)

    N_ = N.compose(N_0)
    assert isinstance(N_, Network)
    assert repr(N_) == "Network('N○N_0')"
