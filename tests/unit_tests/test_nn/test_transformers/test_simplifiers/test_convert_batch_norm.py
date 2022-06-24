import numpy as np

from dnnv.nn import OperationGraph, operations
from dnnv.nn.transformers.simplifiers import simplify
from dnnv.nn.transformers.simplifiers.convert_batch_norm import ConvertBatchNorm
from dnnv.nn.visitors import EnsureSupportVisitor, OperationCounter


def test_convert_batch_norm_on_input():
    input_op = operations.Input((-1, 5), dtype=np.dtype(np.float64))
    bn_op = operations.BatchNormalization(
        input_op,
        np.random.randn(input_op.shape[1]).astype(input_op.dtype),
        np.random.randn(input_op.shape[1]).astype(input_op.dtype),
        np.random.randn(input_op.shape[1]).astype(input_op.dtype),
        abs(np.random.randn(input_op.shape[1]).astype(input_op.dtype)),
    )
    op_graph = OperationGraph([bn_op])
    simplified_op_graph = simplify(op_graph, ConvertBatchNorm(op_graph))

    assert simplified_op_graph.walk(
        EnsureSupportVisitor([operations.Input, operations.Gemm])
    )
    assert op_graph.walk(OperationCounter())[-1] == 2
    assert simplified_op_graph.walk(OperationCounter())[-1] == 2

    x = np.random.randn(100, *input_op.shape[1:]).astype(input_op.dtype)
    y1 = op_graph(x)
    y2 = simplified_op_graph(x)
    assert np.allclose(y1, y2)

    input_op = operations.Input((-1, 3, 5, 5), dtype=np.dtype(np.float64))
    bn_op = operations.BatchNormalization(
        input_op,
        np.random.randn(input_op.shape[1]).astype(input_op.dtype),
        np.random.randn(input_op.shape[1]).astype(input_op.dtype),
        np.random.randn(input_op.shape[1]).astype(input_op.dtype),
        abs(np.random.randn(input_op.shape[1]).astype(input_op.dtype)),
    )
    op_graph = OperationGraph([bn_op])
    simplified_op_graph = simplify(op_graph, ConvertBatchNorm(op_graph))

    assert simplified_op_graph.walk(
        EnsureSupportVisitor([operations.Input, operations.Conv])
    )
    assert op_graph.walk(OperationCounter())[-1] == 2
    assert simplified_op_graph.walk(OperationCounter())[-1] == 2

    x = np.random.randn(100, *input_op.shape[1:]).astype(input_op.dtype)
    y1 = op_graph(x)
    y2 = simplified_op_graph(x)
    assert np.allclose(y1, y2)


def test_convert_batch_norm_on_gemm():
    bn_size = 10
    input_op = operations.Input((-1, 5), dtype=np.dtype(np.float64))
    gemm_op = operations.Gemm(
        input_op,
        np.random.randn(input_op.shape[1], bn_size).astype(input_op.dtype),
    )
    bn_op = operations.BatchNormalization(
        gemm_op,
        np.random.randn(bn_size).astype(input_op.dtype),
        np.random.randn(bn_size).astype(input_op.dtype),
        np.random.randn(bn_size).astype(input_op.dtype),
        abs(np.random.randn(bn_size).astype(input_op.dtype)),
    )
    op_graph = OperationGraph([bn_op])
    simplified_op_graph = simplify(op_graph, ConvertBatchNorm(op_graph))

    assert simplified_op_graph.walk(
        EnsureSupportVisitor([operations.Input, operations.Gemm])
    )

    x = np.random.randn(100, *input_op.shape[1:]).astype(input_op.dtype)
    y1 = op_graph(x)
    y2 = simplified_op_graph(x)
    assert np.allclose(y1, y2)


def test_convert_batch_norm_on_conv():
    bn_size = 5
    input_op = operations.Input((-1, 3, 5, 5), dtype=np.dtype(np.float64))
    conv_op = operations.Conv(
        input_op,
        np.random.randn(bn_size, input_op.shape[1], 2, 2).astype(input_op.dtype),
    )
    bn_op = operations.BatchNormalization(
        conv_op,
        np.random.randn(bn_size).astype(input_op.dtype),
        np.random.randn(bn_size).astype(input_op.dtype),
        np.random.randn(bn_size).astype(input_op.dtype),
        abs(np.random.randn(bn_size).astype(input_op.dtype)),
    )
    op_graph = OperationGraph([bn_op])
    simplified_op_graph = simplify(op_graph, ConvertBatchNorm(op_graph))

    assert simplified_op_graph.walk(
        EnsureSupportVisitor([operations.Input, operations.Conv])
    )

    x = np.random.randn(100, *input_op.shape[1:]).astype(input_op.dtype)
    y1 = op_graph(x)
    y2 = simplified_op_graph(x)
    assert np.allclose(y1, y2)
