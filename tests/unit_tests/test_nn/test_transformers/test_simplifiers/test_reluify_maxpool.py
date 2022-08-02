import numpy as np

from dnnv.nn import OperationGraph, operations
from dnnv.nn.transformers.simplifiers import simplify
from dnnv.nn.transformers.simplifiers.reluify_maxpool import ReluifyMaxPool
from dnnv.nn.visitors import EnsureSupportVisitor


def test_pairwise():
    input_op = operations.Input((-1, 3, 2, 2), dtype=np.dtype(np.float64))
    relu_op = operations.Relu(input_op)
    maxpool_op = operations.MaxPool(relu_op, np.array([1, 2]))
    op_graph = OperationGraph([maxpool_op])
    simplified_op_graph = simplify(op_graph, ReluifyMaxPool(op_graph))

    # only conv and relus should be left after reluification
    assert simplified_op_graph.walk(
        EnsureSupportVisitor([operations.Input, operations.Conv, operations.Relu])
    )

    x = np.random.randn(100, *input_op.shape[1:]).astype(input_op.dtype)
    assert np.allclose(op_graph(x), simplified_op_graph(x))


def test_multistep():
    input_op = operations.Input((-1, 3, 2, 2), dtype=np.dtype(np.float64))
    relu_op = operations.Relu(input_op)
    maxpool_op = operations.MaxPool(relu_op, np.array([2, 2]))
    op_graph = OperationGraph([maxpool_op])
    simplified_op_graph = simplify(op_graph, ReluifyMaxPool(op_graph))

    # only conv and relus should be left after reluification
    assert simplified_op_graph.walk(
        EnsureSupportVisitor([operations.Input, operations.Conv, operations.Relu])
    )

    x = np.random.randn(100, *input_op.shape[1:]).astype(input_op.dtype)
    y1 = op_graph(x)
    y2 = simplified_op_graph(x)
    assert np.allclose(y1, y2)


def test_large_strided():
    input_op = operations.Input((-1, 3, 12, 12), dtype=np.dtype(np.float64))
    relu_op = operations.Relu(input_op)
    maxpool_op = operations.MaxPool(relu_op, np.array([3, 3]), strides=np.array([3, 3]))
    op_graph = OperationGraph([maxpool_op])
    simplified_op_graph = simplify(op_graph, ReluifyMaxPool(op_graph))

    # only conv and relus should be left after reluification
    assert simplified_op_graph.walk(
        EnsureSupportVisitor([operations.Input, operations.Conv, operations.Relu])
    )

    x = np.random.randn(100, *input_op.shape[1:]).astype(input_op.dtype)
    y1 = op_graph(x)
    y2 = simplified_op_graph(x)
    assert np.allclose(y1, y2)


def test_large_padded():
    input_op = operations.Input((-1, 3, 12, 12), dtype=np.dtype(np.float64))
    relu_op = operations.Relu(input_op)
    maxpool_op = operations.MaxPool(
        relu_op, np.array([3, 3]), pads=np.array([1, 1, 1, 1])
    )
    op_graph = OperationGraph([maxpool_op])
    simplified_op_graph = simplify(op_graph, ReluifyMaxPool(op_graph))

    # only conv and relus should be left after reluification
    assert simplified_op_graph.walk(
        EnsureSupportVisitor([operations.Input, operations.Conv, operations.Relu])
    )

    x = np.random.randn(100, *input_op.shape[1:]).astype(input_op.dtype)
    y1 = op_graph(x)
    y2 = simplified_op_graph(x)
    assert np.allclose(y1, y2)


def test_large_padded_and_strided():
    input_op = operations.Input((-1, 3, 24, 18), dtype=np.dtype(np.float64))
    relu_op = operations.Relu(input_op)
    maxpool_op = operations.MaxPool(
        relu_op, np.array([3, 3]), strides=np.array([2, 2]), pads=np.array([1, 1, 1, 1])
    )
    op_graph = OperationGraph([maxpool_op])
    simplified_op_graph = simplify(op_graph, ReluifyMaxPool(op_graph))

    # only conv and relus should be left after reluification
    assert simplified_op_graph.walk(
        EnsureSupportVisitor([operations.Input, operations.Conv, operations.Relu])
    )

    x = np.random.randn(100, *input_op.shape[1:]).astype(input_op.dtype)
    y1 = op_graph(x)
    y2 = simplified_op_graph(x)
    assert np.allclose(y1, y2)


def test_large_padded_and_strided_relu_first():
    input_op = operations.Input((-1, 3, 24, 18), dtype=np.dtype(np.float64))
    maxpool_op = operations.MaxPool(
        input_op,
        np.array([3, 3]),
        strides=np.array([2, 2]),
        pads=np.array([1, 1, 1, 1]),
    )
    relu_op = operations.Relu(maxpool_op)
    op_graph = OperationGraph([relu_op])
    simplified_op_graph = simplify(op_graph, ReluifyMaxPool(op_graph))

    # only conv and relus should be left after reluification
    assert simplified_op_graph.walk(
        EnsureSupportVisitor([operations.Input, operations.Conv, operations.Relu])
    )

    x = np.random.randn(100, *input_op.shape[1:]).astype(input_op.dtype)
    y1 = op_graph(x)
    y2 = simplified_op_graph(x)
    assert np.allclose(y1, y2)
