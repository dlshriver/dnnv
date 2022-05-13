import numpy as np

from dnnv.nn import OperationGraph, operations
from dnnv.nn.transformers.simplifiers import simplify
from dnnv.nn.transformers.simplifiers.squeeze_gemms import SqueezeGemms
from dnnv.nn.visitors import EnsureSupportVisitor, OperationCounter


def test_squeeze_gemms_no_c():
    input_op = operations.Input((-1, 5), dtype=np.dtype(np.float64))
    gemm_op_1 = operations.Gemm(
        input_op, np.random.randn(input_op.shape[1], 10).astype(input_op.dtype)
    )
    gemm_op_2 = operations.Gemm(
        gemm_op_1, np.random.randn(10, 10).astype(input_op.dtype)
    )
    op_graph = OperationGraph([gemm_op_2])
    simplified_op_graph = simplify(op_graph, SqueezeGemms())

    assert EnsureSupportVisitor([operations.Input, operations.Gemm]).walk(
        simplified_op_graph
    )
    assert OperationCounter().walk(op_graph) == 3
    assert OperationCounter().walk(simplified_op_graph) == 2

    x = np.random.randn(100, *input_op.shape[1:]).astype(input_op.dtype)
    y1 = op_graph(x)
    y2 = simplified_op_graph(x)
    assert np.allclose(y1, y2)


def test_squeeze_gemms_one_c():
    input_op = operations.Input((-1, 5), dtype=np.dtype(np.float64))
    gemm_op_1 = operations.Gemm(
        input_op,
        np.random.randn(input_op.shape[1], 10).astype(input_op.dtype),
        np.random.randn(10).astype(input_op.dtype),
    )
    gemm_op_2 = operations.Gemm(
        gemm_op_1, np.random.randn(10, 10).astype(input_op.dtype)
    )
    op_graph = OperationGraph([gemm_op_2])
    simplified_op_graph = simplify(op_graph, SqueezeGemms())

    assert EnsureSupportVisitor([operations.Input, operations.Gemm]).walk(
        simplified_op_graph
    )
    assert OperationCounter().walk(op_graph) == 3
    assert OperationCounter().walk(simplified_op_graph) == 2

    x = np.random.randn(100, *input_op.shape[1:]).astype(input_op.dtype)
    y1 = op_graph(x)
    y2 = simplified_op_graph(x)
    assert np.allclose(y1, y2)

    input_op = operations.Input((-1, 5), dtype=np.dtype(np.float64))
    gemm_op_1 = operations.Gemm(
        input_op, np.random.randn(input_op.shape[1], 10).astype(input_op.dtype)
    )
    gemm_op_2 = operations.Gemm(
        gemm_op_1,
        np.random.randn(10, 10).astype(input_op.dtype),
        np.random.randn(10).astype(input_op.dtype),
    )
    op_graph = OperationGraph([gemm_op_2])
    simplified_op_graph = simplify(op_graph, SqueezeGemms())

    assert EnsureSupportVisitor([operations.Input, operations.Gemm]).walk(
        simplified_op_graph
    )
    assert OperationCounter().walk(op_graph) == 3
    assert OperationCounter().walk(simplified_op_graph) == 2

    x = np.random.randn(100, *input_op.shape[1:]).astype(input_op.dtype)
    y1 = op_graph(x)
    y2 = simplified_op_graph(x)
    assert np.allclose(y1, y2)


def test_squeeze_gemms_with_conv():
    input_op = operations.Input((-1, 3, 4, 4), dtype=np.dtype(np.float64))
    conv_op = operations.Conv(
        input_op,
        np.random.randn(3, 3, 1, 1).astype(input_op.dtype),
    )
    flatten_op = operations.Flatten(conv_op)
    gemm_op = operations.Gemm(
        flatten_op,
        np.random.randn(np.product(input_op.shape[1:]), 10).astype(input_op.dtype),
    )
    op_graph = OperationGraph([gemm_op])
    simplified_op_graph = simplify(op_graph, SqueezeGemms())

    assert EnsureSupportVisitor(
        [operations.Input, operations.Flatten, operations.Gemm]
    ).walk(simplified_op_graph)
    assert OperationCounter().walk(op_graph) == 4
    assert OperationCounter().walk(simplified_op_graph) == 3

    x = np.random.randn(100, *input_op.shape[1:]).astype(input_op.dtype)
    y1 = op_graph(x)
    y2 = simplified_op_graph(x)
    assert np.allclose(y1, y2)


def test_squeeze_gemms_with_conv_with_bias():
    input_op = operations.Input((-1, 3, 4, 4), dtype=np.dtype(np.float64))
    conv_op = operations.Conv(
        input_op,
        np.random.randn(3, 3, 1, 1).astype(input_op.dtype),
    )
    flatten_op = operations.Flatten(conv_op)
    gemm_op = operations.Gemm(
        flatten_op,
        np.random.randn(np.product(input_op.shape[1:]), 10).astype(input_op.dtype),
        np.random.randn(10).astype(input_op.dtype),
    )
    op_graph = OperationGraph([gemm_op])
    simplified_op_graph = simplify(op_graph, SqueezeGemms())

    assert EnsureSupportVisitor(
        [operations.Input, operations.Flatten, operations.Gemm]
    ).walk(simplified_op_graph)
    assert OperationCounter().walk(op_graph) == 4
    assert OperationCounter().walk(simplified_op_graph) == 3

    x = np.random.randn(100, *input_op.shape[1:]).astype(input_op.dtype)
    y1 = op_graph(x)
    y2 = simplified_op_graph(x)
    assert np.allclose(y1, y2)

    input_op = operations.Input((-1, 3, 4, 4), dtype=np.dtype(np.float64))
    conv_op = operations.Conv(
        input_op,
        np.random.randn(3, 3, 1, 1).astype(input_op.dtype),
        np.random.randn(3).astype(input_op.dtype),
    )
    flatten_op = operations.Flatten(conv_op)
    gemm_op = operations.Gemm(
        flatten_op,
        np.random.randn(np.product(input_op.shape[1:]), 10).astype(input_op.dtype),
        np.random.randn(10).astype(input_op.dtype),
    )
    op_graph = OperationGraph([gemm_op])
    simplified_op_graph = simplify(op_graph, SqueezeGemms())

    assert EnsureSupportVisitor(
        [operations.Input, operations.Flatten, operations.Gemm]
    ).walk(simplified_op_graph)
    assert OperationCounter().walk(op_graph) == 4
    assert OperationCounter().walk(simplified_op_graph) == 3

    x = np.random.randn(100, *input_op.shape[1:]).astype(input_op.dtype)
    y1 = op_graph(x)
    y2 = simplified_op_graph(x)
    assert np.allclose(y1, y2)
