import numpy as np

from dnnv.nn import OperationGraph, operations
from dnnv.nn.transformers.simplifiers import simplify
from dnnv.nn.transformers.simplifiers.bundle_padding import BundlePadding
from dnnv.nn.visitors import EnsureSupportVisitor, OperationCounter


def test_bundle_padding_conv():
    input_op = operations.Input((-1, 3, 2, 2), dtype=np.dtype(np.float64))
    pad_op = operations.Pad(input_op, (0, 0, 2, 2, 0, 0, 2, 2))
    conv_op = operations.Conv(
        pad_op, np.random.randn(5, 3, 2, 2).astype(input_op.dtype)
    )
    op_graph = OperationGraph([conv_op])
    simplified_op_graph = simplify(op_graph, BundlePadding(op_graph))

    assert simplified_op_graph.walk(
        EnsureSupportVisitor([operations.Input, operations.Conv])
    )
    assert op_graph.walk(OperationCounter())[-1] == 3
    assert simplified_op_graph.walk(OperationCounter())[-1] == 2

    x = np.random.randn(100, *input_op.shape[1:]).astype(input_op.dtype)
    y1 = op_graph(x)
    y2 = simplified_op_graph(x)
    assert np.allclose(y1, y2)


def test_bundle_padding_conv_nopad():
    input_op = operations.Input((-1, 3, 2, 2), dtype=np.dtype(np.float64))
    pad_op = operations.Pad(input_op, (0, 0, 0, 0, 0, 0, 0, 0))
    conv_op = operations.Conv(
        pad_op, np.random.randn(5, 3, 2, 2).astype(input_op.dtype)
    )
    op_graph = OperationGraph([conv_op])
    simplified_op_graph = simplify(op_graph, BundlePadding(op_graph))

    assert simplified_op_graph.walk(
        EnsureSupportVisitor([operations.Input, operations.Conv])
    )
    assert op_graph.walk(OperationCounter())[-1] == 3
    assert simplified_op_graph.walk(OperationCounter())[-1] == 2

    x = np.random.randn(100, *input_op.shape[1:]).astype(input_op.dtype)
    y1 = op_graph(x)
    y2 = simplified_op_graph(x)
    assert np.allclose(y1, y2)


def test_bundle_padding_conv_noop():
    input_op = operations.Input((-1, 3, 2, 2), dtype=np.dtype(np.float64))
    pad_op = operations.Pad(input_op, (0, 1, 2, 2, 0, 1, 2, 2))
    conv_op = operations.Conv(
        pad_op, np.random.randn(5, 3, 2, 2).astype(input_op.dtype)
    )
    # testing weird padding
    op_graph = OperationGraph([conv_op])
    simplified_op_graph = simplify(op_graph, BundlePadding(op_graph))
    assert simplified_op_graph.output_operations[0] == op_graph.output_operations[0]

    input_op = operations.Input((-1, 3, 2, 2), dtype=np.dtype(np.float64))
    pad_op = operations.Pad(input_op, (0, 0, 2, 2, 0, 0, 2, 2), mode="reflect")
    conv_op = operations.Conv(
        pad_op, np.random.randn(5, 3, 2, 2).astype(input_op.dtype)
    )
    # testing non-constant padding
    op_graph = OperationGraph([conv_op])
    simplified_op_graph = simplify(op_graph, BundlePadding(op_graph))
    assert simplified_op_graph.output_operations[0] == op_graph.output_operations[0]

    input_op = operations.Input((-1, 3, 2, 2), dtype=np.dtype(np.float64))
    pad_op = operations.Pad(input_op, (0, 0, 2, 2, 0, 0, 2, 2), value=5)
    conv_op = operations.Conv(
        pad_op, np.random.randn(5, 3, 2, 2).astype(input_op.dtype)
    )
    # testing non-zero constant padding
    op_graph = OperationGraph([conv_op])
    simplified_op_graph = simplify(op_graph, BundlePadding(op_graph))
    assert simplified_op_graph.output_operations[0] == op_graph.output_operations[0]


def test_bundle_padding_maxpool():
    input_op = operations.Input((-1, 3, 2, 2), dtype=np.dtype(np.float64))
    pad_op = operations.Pad(input_op, (0, 0, 2, 2, 0, 0, 2, 2), value=np.nan)
    maxpool_op = operations.MaxPool(pad_op, (2, 2))
    op_graph = OperationGraph([maxpool_op])
    simplified_op_graph = simplify(op_graph, BundlePadding(op_graph))

    assert simplified_op_graph.walk(
        EnsureSupportVisitor([operations.Input, operations.MaxPool])
    )
    assert op_graph.walk(OperationCounter())[-1] == 3
    assert simplified_op_graph.walk(OperationCounter())[-1] == 2

    x = np.random.randn(100, *input_op.shape[1:]).astype(input_op.dtype)
    y1 = op_graph(x)
    y2 = simplified_op_graph(x)
    assert np.allclose(y1, y2)


def test_bundle_padding_maxpool_nopad():
    input_op = operations.Input((-1, 3, 2, 2), dtype=np.dtype(np.float64))
    pad_op = operations.Pad(input_op, (0, 0, 0, 0, 0, 0, 0, 0))
    maxpool_op = operations.MaxPool(pad_op, (2, 2))
    op_graph = OperationGraph([maxpool_op])
    simplified_op_graph = simplify(op_graph, BundlePadding(op_graph))

    assert simplified_op_graph.walk(
        EnsureSupportVisitor([operations.Input, operations.MaxPool])
    )
    assert op_graph.walk(OperationCounter())[-1] == 3
    assert simplified_op_graph.walk(OperationCounter())[-1] == 2

    x = np.random.randn(100, *input_op.shape[1:]).astype(input_op.dtype)
    y1 = op_graph(x)
    y2 = simplified_op_graph(x)
    assert np.allclose(y1, y2)


def test_bundle_padding_maxpool_noop():
    input_op = operations.Input((-1, 3, 2, 2), dtype=np.dtype(np.float64))
    pad_op = operations.Pad(input_op, (0, 1, 2, 2, 0, 1, 2, 2))
    maxpool_op = operations.MaxPool(pad_op, (2, 2))
    # testing weird padding
    op_graph = OperationGraph([maxpool_op])
    simplified_op_graph = simplify(op_graph, BundlePadding(op_graph))
    assert simplified_op_graph.output_operations[0] == op_graph.output_operations[0]

    input_op = operations.Input((-1, 3, 2, 2), dtype=np.dtype(np.float64))
    pad_op = operations.Pad(input_op, (0, 0, 2, 2, 0, 0, 2, 2), mode="reflect")
    maxpool_op = operations.MaxPool(pad_op, (2, 2))
    # testing non-constant padding
    op_graph = OperationGraph([maxpool_op])
    simplified_op_graph = simplify(op_graph, BundlePadding(op_graph))
    assert simplified_op_graph.output_operations[0] == op_graph.output_operations[0]

    input_op = operations.Input((-1, 3, 2, 2), dtype=np.dtype(np.float64))
    pad_op = operations.Pad(input_op, (0, 0, 2, 2, 0, 0, 2, 2), value=5)
    maxpool_op = operations.MaxPool(pad_op, (2, 2))
    # testing non-nan constant padding
    op_graph = OperationGraph([maxpool_op])
    simplified_op_graph = simplify(op_graph, BundlePadding(op_graph))
    assert simplified_op_graph.output_operations[0] == op_graph.output_operations[0]
