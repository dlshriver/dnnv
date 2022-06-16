import numpy as np

from dnnv.nn import OperationGraph, operations
from dnnv.nn.transformers.simplifiers import simplify
from dnnv.nn.transformers.simplifiers.drop_identities import (
    DropDropout,
    DropIdentity,
    DropUnnecessaryConcat,
    DropUnnecessaryFlatten,
    DropUnnecessaryRelu,
)
from dnnv.nn.visitors import EnsureSupportVisitor, OperationCounter


def test_drop_dropout():
    input_op = operations.Input((-1, 3, 2, 2), dtype=np.dtype(np.float64))
    dropout_op = operations.Dropout(input_op)
    op_graph = OperationGraph([dropout_op])
    simplified_op_graph = simplify(op_graph, DropDropout())

    assert EnsureSupportVisitor([operations.Input]).walk(simplified_op_graph)

    x = np.random.randn(100, *input_op.shape[1:]).astype(input_op.dtype)
    y1 = op_graph(x)
    y2 = simplified_op_graph(x)
    assert np.allclose(y1, y2)


def test_drop_identity():
    input_op = operations.Input((-1, 3, 2, 2), dtype=np.dtype(np.float64))
    identity_op = operations.Identity(input_op)
    op_graph = OperationGraph([identity_op])
    simplified_op_graph = simplify(op_graph, DropIdentity())

    assert EnsureSupportVisitor([operations.Input]).walk(simplified_op_graph)

    x = np.random.randn(100, *input_op.shape[1:]).astype(input_op.dtype)
    y1 = op_graph(x)
    y2 = simplified_op_graph(x)
    assert np.allclose(y1, y2)


def test_drop_unnecessary_concat():
    input_op = operations.Input((-1, 5), dtype=np.dtype(np.float64))
    identity_op = operations.Concat([input_op], axis=1)
    op_graph = OperationGraph([identity_op])
    simplified_op_graph = simplify(op_graph, DropUnnecessaryConcat())

    assert EnsureSupportVisitor([operations.Input]).walk(simplified_op_graph)

    x = np.random.randn(100, *input_op.shape[1:]).astype(input_op.dtype)
    y1 = op_graph(x)
    y2 = simplified_op_graph(x)
    assert np.allclose(y1, y2)


def test_drop_unnecessary_flatten():
    input_op = operations.Input((-1, 5), dtype=np.dtype(np.float64))
    identity_op = operations.Flatten(input_op, axis=1)
    op_graph = OperationGraph([identity_op])
    simplified_op_graph = simplify(op_graph, DropUnnecessaryFlatten())

    assert EnsureSupportVisitor([operations.Input]).walk(simplified_op_graph)

    x = np.random.randn(100, *input_op.shape[1:]).astype(input_op.dtype)
    y1 = op_graph(x)
    y2 = simplified_op_graph(x)
    assert np.allclose(y1, y2)


def test_drop_unnecessary_relu():
    input_op = operations.Input((-1, 5), dtype=np.dtype(np.float64))
    relu_op = operations.Relu(input_op)
    identity_op = operations.Relu(relu_op)
    op_graph = OperationGraph([identity_op])
    simplified_op_graph = simplify(op_graph, DropUnnecessaryRelu())

    assert OperationCounter().walk(simplified_op_graph) == 2

    x = np.random.randn(100, *input_op.shape[1:]).astype(input_op.dtype)
    y1 = op_graph(x)
    y2 = simplified_op_graph(x)
    assert np.allclose(y1, y2)

    input_op = operations.Input((-1, 5), dtype=np.dtype(np.float64))
    relu_op = operations.Sigmoid(input_op)
    identity_op = operations.Relu(relu_op)
    op_graph = OperationGraph([identity_op])
    simplified_op_graph = simplify(op_graph, DropUnnecessaryRelu())

    assert EnsureSupportVisitor([operations.Input, operations.Sigmoid]).walk(
        simplified_op_graph
    )
    assert OperationCounter().walk(simplified_op_graph) == 2

    x = np.random.randn(100, *input_op.shape[1:]).astype(input_op.dtype)
    y1 = op_graph(x)
    y2 = simplified_op_graph(x)
    assert np.allclose(y1, y2)
