import numpy as np
import onnx
import pytest

from dnnv.nn.graph import OperationGraph
from dnnv.nn.layers import Convolutional
from dnnv.nn.operations import *


def test_from_operation_graph_error():
    op = Input((-1, 5), np.dtype(np.float32))
    cast_op = Cast(op, onnx.TensorProto.FLOAT)
    op_graph = OperationGraph([cast_op])
    with pytest.raises(ValueError) as excinfo:
        layer = Convolutional.from_operation_graph(op_graph)
    assert str(excinfo.value).startswith(
        "Expected operation of type (Conv | Activation)"
    )

    op_graph = OperationGraph([Relu(cast_op)])
    with pytest.raises(ValueError) as excinfo:
        layer = Convolutional.from_operation_graph(op_graph)
    assert str(excinfo.value).startswith("Expected type Conv")

    W = np.ones((3, 5, 2, 2), dtype=np.float32)
    b = np.zeros(2, dtype=np.float32)
    conv_op = Conv(op, W, b, dilations=np.array([2, 2]))
    op_graph = OperationGraph([conv_op])
    with pytest.raises(ValueError) as excinfo:
        layer = Convolutional.from_operation_graph(op_graph)
    assert str(excinfo.value).startswith(
        "Dilation is currently not supported in Convolutional layers."
    )

    conv_op = Conv(op, W, b, group=2)
    op_graph = OperationGraph([conv_op])
    with pytest.raises(ValueError) as excinfo:
        layer = Convolutional.from_operation_graph(op_graph)
    assert str(excinfo.value).startswith(
        "Grouping is currently not supported in Convolutional layers."
    )

    conv_op = Conv(np.ones((1, 3, 2, 2), dtype=np.float32), W, b)
    op_graph = OperationGraph([conv_op])
    with pytest.raises(ValueError) as excinfo:
        layer = Convolutional.from_operation_graph(op_graph)
    assert str(excinfo.value).startswith(
        "Constant input tensors are not supported for Conv"
    )


def test_from_operation_graph_0():
    op = Input((-1, 3, 2, 2), np.dtype(np.float32))
    W = np.ones((3, 5, 2, 2), dtype=np.float32)
    b = np.zeros(2, dtype=np.float32)
    conv_op = Conv(op, W, b)
    op_graph = OperationGraph([conv_op])
    layer = Convolutional.from_operation_graph(op_graph)
    assert isinstance(layer, Convolutional)
    assert layer.activation is None
    assert np.allclose(layer.weights, W)
    assert np.allclose(layer.bias, b)
    assert np.allclose(layer.kernel_shape, np.array([2, 2]))
    assert np.allclose(layer.strides, np.array([1, 1]))
    assert np.allclose(layer.pads, np.array([0, 0, 0, 0]))


def test_from_operation_graph_1():
    op = Input((-1, 3, 2, 2), np.dtype(np.float32))
    W = np.ones((3, 5, 2, 2), dtype=np.float32)
    b = np.zeros(2, dtype=np.float32)
    conv_op = Conv(op, W, b)
    relu_op = Relu(conv_op)
    op_graph = OperationGraph([relu_op])
    layer = Convolutional.from_operation_graph(op_graph)
    assert isinstance(layer, Convolutional)
    assert layer.activation == "relu"
    assert np.allclose(layer.weights, W)
    assert np.allclose(layer.bias, b)
    assert np.allclose(layer.kernel_shape, np.array([2, 2]))
    assert np.allclose(layer.strides, np.array([1, 1]))
    assert np.allclose(layer.pads, np.array([0, 0, 0, 0]))

    sigmoid_op = Sigmoid(conv_op)
    op_graph = OperationGraph([sigmoid_op])
    layer = Convolutional.from_operation_graph(op_graph)
    assert isinstance(layer, Convolutional)
    assert layer.activation == "sigmoid"
    assert np.allclose(layer.weights, W)
    assert np.allclose(layer.bias, b)
    assert np.allclose(layer.kernel_shape, np.array([2, 2]))
    assert np.allclose(layer.strides, np.array([1, 1]))
    assert np.allclose(layer.pads, np.array([0, 0, 0, 0]))


def test_from_operation_graph_2():
    op = Input((-1, 3, 2, 2), np.dtype(np.float32))
    W = np.ones((3, 5, 2, 2), dtype=np.float32)
    b = np.zeros(2, dtype=np.float32)
    conv_op = Conv(op, W, b)
    conv_op.kernel_shape = (
        None  # TODO : Why do this? Maybe that check should be removed
    )
    op_graph = OperationGraph([conv_op])
    layer = Convolutional.from_operation_graph(op_graph)
    assert isinstance(layer, Convolutional)
    assert layer.activation is None
    assert np.allclose(layer.weights, W)
    assert np.allclose(layer.bias, b)
    assert np.allclose(layer.kernel_shape, np.array([2, 2]))
    assert np.allclose(layer.strides, np.array([1, 1]))
    assert np.allclose(layer.pads, np.array([0, 0, 0, 0]))
