import numpy as np
import onnx
import pytest

from dnnv.nn.graph import OperationGraph
from dnnv.nn.layers import FullyConnected
from dnnv.nn.operations import *


def test_from_operation_graph_error():
    op = Input((-1, 5), np.dtype(np.float32))
    cast_op = Cast(op, onnx.TensorProto.FLOAT)
    op_graph = OperationGraph([cast_op])
    with pytest.raises(ValueError) as excinfo:
        layer = FullyConnected.from_operation_graph(op_graph)
    assert str(excinfo.value).startswith(
        "Expected operation of type (Gemm | Add | Activation)"
    )

    op = Input((-1, 5), np.dtype(np.float32))
    W = np.ones((5, 2), dtype=np.float32)
    b = np.zeros(2, dtype=np.float32)
    gemm_op = Gemm(op, W, b, alpha=2.0)
    op_graph = OperationGraph([gemm_op])
    with pytest.raises(ValueError) as excinfo:
        layer = FullyConnected.from_operation_graph(op_graph)
    assert str(excinfo.value).startswith(
        "Scaling not supported in Fully Connected layers."
    )
    gemm_op = Gemm(op, W, b, beta=2.0)
    op_graph = OperationGraph([gemm_op])
    with pytest.raises(ValueError) as excinfo:
        layer = FullyConnected.from_operation_graph(op_graph)
    assert str(excinfo.value).startswith(
        "Scaling not supported in Fully Connected layers."
    )

    gemm_op = Gemm(np.ones((1, 5), dtype=np.float32), W, b)
    op_graph = OperationGraph([gemm_op])
    with pytest.raises(ValueError) as excinfo:
        layer = FullyConnected.from_operation_graph(op_graph)
    assert str(excinfo.value).startswith("Constant input tensors are not supported")

    gemm_op = Gemm(op, W, b, transpose_a=True)
    op_graph = OperationGraph([gemm_op])
    with pytest.raises(ValueError) as excinfo:
        layer = FullyConnected.from_operation_graph(op_graph)
    assert str(excinfo.value).startswith(
        "Transposing input to Fully Connected layer is not supported."
    )

    input_op = Input((5, 2), np.dtype(np.float32))
    gemm_op = Gemm(op, input_op, b)
    op_graph = OperationGraph([gemm_op])
    with pytest.raises(ValueError) as excinfo:
        layer = FullyConnected.from_operation_graph(op_graph)
    assert str(excinfo.value).startswith("Multiple input tensors are not supported")

    input_op = Input((2,), np.dtype(np.float32))
    gemm_op = Gemm(op, W, input_op)
    op_graph = OperationGraph([gemm_op])
    with pytest.raises(ValueError) as excinfo:
        layer = FullyConnected.from_operation_graph(op_graph)
    assert str(excinfo.value).startswith(
        "Variable input tensors are not supported for GeMM bias"
    )

    relu_op = Relu(op)
    op_graph = OperationGraph([relu_op])
    with pytest.raises(ValueError) as excinfo:
        layer = FullyConnected.from_operation_graph(op_graph)
    assert str(excinfo.value).startswith("Expected type (Gemm | (MatMul >> Add))")

    matmul_op = MatMul(op, W)
    add_op = Add(b, matmul_op)
    op_graph = OperationGraph([add_op])
    with pytest.raises(ValueError) as excinfo:
        layer = FullyConnected.from_operation_graph(op_graph)
    assert str(excinfo.value).startswith(
        "Constant input tensors are not supported for Add in Fully Connected layers."
    )

    x = np.ones((1, 5), dtype=np.float32)
    matmul_op = MatMul(x, W)
    add_op = Add(matmul_op, b)
    op_graph = OperationGraph([add_op])
    with pytest.raises(ValueError) as excinfo:
        layer = FullyConnected.from_operation_graph(op_graph)
    assert str(excinfo.value).startswith(
        "Constant input tensors are not supported for MatMul in Fully Connected layers."
    )

    atan_op = Atan(op)
    gemm_op = Gemm(atan_op, W, b)
    op_graph = OperationGraph([gemm_op])
    with pytest.raises(ValueError) as excinfo:
        layer = FullyConnected.from_operation_graph(op_graph)
    assert str(excinfo.value).startswith(
        "Expected type (None | (Transpose >> (Flatten | Reshape)))"
    )

    atan_op = Atan(op)
    flatten_op = Flatten(atan_op)
    gemm_op = Gemm(flatten_op, W, b)
    op_graph = OperationGraph([gemm_op])
    with pytest.raises(ValueError) as excinfo:
        layer = FullyConnected.from_operation_graph(op_graph)
    assert str(excinfo.value).startswith("Expected type Transpose")

    W = np.ones((2, 5), dtype=np.float32)
    b = np.zeros(2, dtype=np.float32)
    atan_op = Atan(op)
    transpose_op = Transpose(atan_op, permutation=np.array([0, 3, 1, 2]))
    reshape_op = Reshape(transpose_op, np.array([-1, 12]))
    gemm_op = Gemm(reshape_op, W, b)
    op_graph = OperationGraph([gemm_op])
    with pytest.raises(ValueError) as excinfo:
        layer = FullyConnected.from_operation_graph(op_graph)
    assert str(excinfo.value).startswith("Expected Transpose to be applied to Input.")


def test_from_operation_graph_0():
    op = Input((-1, 5), np.dtype(np.float32))
    W = np.ones((5, 2), dtype=np.float32)
    b = np.zeros(2, dtype=np.float32)
    gemm_op = Gemm(op, W, b)
    op_graph = OperationGraph([gemm_op])
    layer = FullyConnected.from_operation_graph(op_graph)
    assert isinstance(layer, FullyConnected)
    assert layer.activation is None
    assert np.allclose(layer.weights, W)
    assert np.allclose(layer.bias, b)


def test_from_operation_graph_1():
    op = Input((-1, 5), np.dtype(np.float32))
    W = np.ones((5, 2), dtype=np.float32)
    b = np.zeros(2, dtype=np.float32)
    gemm_op = Gemm(op, W, b)
    relu_op = Relu(gemm_op)
    op_graph = OperationGraph([relu_op])
    layer = FullyConnected.from_operation_graph(op_graph)
    assert isinstance(layer, FullyConnected)
    assert layer.activation == "relu"
    assert np.allclose(layer.weights, W)
    assert np.allclose(layer.bias, b)


def test_from_operation_graph_2():
    op = Input((-1, 5), np.dtype(np.float32))
    W = np.ones((5, 2), dtype=np.float32)
    b = np.zeros(2, dtype=np.float32)
    matmul_op = MatMul(op, W)
    add_op = Add(matmul_op, b)
    relu_op = Relu(add_op)
    op_graph = OperationGraph([relu_op])
    layer = FullyConnected.from_operation_graph(op_graph)
    assert isinstance(layer, FullyConnected)
    assert layer.activation == "relu"
    assert np.allclose(layer.weights, W)
    assert np.allclose(layer.bias, b)


def test_from_operation_graph_3():
    op = Input((-1, 5), np.dtype(np.float32))
    W = np.ones((2, 5), dtype=np.float32)
    b = np.zeros(2, dtype=np.float32)
    gemm_op = Gemm(op, W, b, transpose_b=True)
    relu_op = Relu(gemm_op)
    op_graph = OperationGraph([relu_op])
    layer = FullyConnected.from_operation_graph(op_graph)
    assert isinstance(layer, FullyConnected)
    assert layer.activation == "relu"
    assert np.allclose(layer.weights, W.T)
    assert np.allclose(layer.bias, b)


def test_from_operation_graph_4():
    op = Input((-1, 5), np.dtype(np.float32))
    W = np.ones((2, 5), dtype=np.float32)
    b = np.zeros(2, dtype=np.float32)
    flatten_op = Flatten(op)
    gemm_op = Gemm(flatten_op, W, b)
    op_graph = OperationGraph([gemm_op])
    layer = FullyConnected.from_operation_graph(op_graph)
    assert isinstance(layer, FullyConnected)
    assert layer.activation is None
    assert np.allclose(layer.weights, W)
    assert np.allclose(layer.bias, b)


def test_from_operation_graph_5():
    op = Input((-1, 2, 2, 3), np.dtype(np.float32))
    W = np.ones((2, 5), dtype=np.float32)
    b = np.zeros(2, dtype=np.float32)
    transpose_op = Transpose(op, permutation=np.array([0, 3, 1, 2]))
    reshape_op = Reshape(transpose_op, np.array([-1, 12]))
    gemm_op = Gemm(reshape_op, W, b)
    op_graph = OperationGraph([gemm_op])
    layer = FullyConnected.from_operation_graph(op_graph)
    assert isinstance(layer, FullyConnected)
    assert layer.activation is None
    assert np.allclose(layer.weights, W)
    assert np.allclose(layer.bias, b)
    # assert np.allclose(layer.w_permutation, W)
