import numpy as np

from dnnv.nn import OperationGraph, operations
from dnnv.nn.transformers.simplifiers import simplify
from dnnv.nn.transformers.simplifiers.convert_mul import ConvertMul
from dnnv.nn.visitors import EnsureSupportVisitor


def test_convert_mul():
    input_op = operations.Input((-1, 5), dtype=np.dtype(np.float64))
    mul_op = operations.Mul(
        input_op, np.random.randn(1, *input_op.shape[1:]).astype(input_op.dtype)
    )
    op_graph = OperationGraph([mul_op])
    simplified_op_graph = simplify(op_graph, ConvertMul(op_graph))

    assert simplified_op_graph.walk(
        EnsureSupportVisitor(
            [
                operations.Input,
                operations.Add,
                operations.MatMul,
                operations.Gemm,
                operations.Conv,
            ]
        )
    )

    x = np.random.randn(100, *input_op.shape[1:]).astype(input_op.dtype)
    y1 = op_graph(x)
    y2 = simplified_op_graph(x)
    assert np.allclose(y1, y2)

    mul_op = operations.Mul(
        np.random.randn(1, *input_op.shape[1:]).astype(input_op.dtype), input_op
    )
    op_graph = OperationGraph([mul_op])
    simplified_op_graph = simplify(op_graph, ConvertMul(op_graph))

    assert simplified_op_graph.walk(
        EnsureSupportVisitor(
            [
                operations.Input,
                operations.Add,
                operations.MatMul,
                operations.Gemm,
                operations.Conv,
            ]
        )
    )

    x = np.random.randn(100, *input_op.shape[1:]).astype(input_op.dtype)
    y1 = op_graph(x)
    y2 = simplified_op_graph(x)
    assert np.allclose(y1, y2)


def test_convert_mul_ones():
    input_op = operations.Input((-1, 3, 4, 5), dtype=np.dtype(np.float64))
    mul_op = operations.Mul(
        input_op, np.ones((1, *input_op.shape[1:]), dtype=input_op.dtype)
    )
    op_graph = OperationGraph([mul_op])
    simplified_op_graph = simplify(op_graph, ConvertMul(op_graph))

    assert simplified_op_graph.walk(EnsureSupportVisitor([operations.Input]))

    x = np.random.randn(100, *input_op.shape[1:]).astype(input_op.dtype)
    y1 = op_graph(x)
    y2 = simplified_op_graph(x)
    assert np.allclose(y1, y2)

    mul_op = operations.Mul(
        np.ones((1, *input_op.shape[1:]), dtype=input_op.dtype), input_op
    )
    op_graph = OperationGraph([mul_op])
    simplified_op_graph = simplify(op_graph, ConvertMul(op_graph))

    assert simplified_op_graph.walk(EnsureSupportVisitor([operations.Input]))

    x = np.random.randn(100, *input_op.shape[1:]).astype(input_op.dtype)
    y1 = op_graph(x)
    y2 = simplified_op_graph(x)
    assert np.allclose(y1, y2)


def test_convert_mul_zero():
    input_op = operations.Input((-1, 3, 4, 5), dtype=np.dtype(np.float64))
    mul_op = operations.Mul(
        input_op, np.zeros((1, *input_op.shape[1:]), dtype=input_op.dtype)
    )
    op_graph = OperationGraph([mul_op])
    simplified_op_graph = simplify(op_graph, ConvertMul(op_graph))
    assert np.allclose(
        simplified_op_graph.output_operations[0], np.zeros((1, *input_op.shape[1:]))
    )

    mul_op = operations.Mul(
        np.zeros((1, *input_op.shape[1:]), dtype=input_op.dtype), input_op
    )
    op_graph = OperationGraph([mul_op])
    simplified_op_graph = simplify(op_graph, ConvertMul(op_graph))
    assert np.allclose(
        simplified_op_graph.output_operations[0], np.zeros((1, *input_op.shape[1:]))
    )


def test_convert_mul_constants():
    shape = (1, 5)
    dtype = np.float64
    mul_op = operations.Mul(
        np.random.randn(*shape).astype(dtype), np.random.randn(*shape).astype(dtype)
    )
    op_graph = OperationGraph([mul_op])
    simplified_op_graph = simplify(op_graph, ConvertMul(op_graph))
    assert isinstance(simplified_op_graph.output_operations[0], np.ndarray)


def test_convert_mul_1d():
    input_op = operations.Input((5,), dtype=np.dtype(np.float64))
    mul_op = operations.Mul(
        input_op, np.random.randn(*input_op.shape).astype(input_op.dtype)
    )
    op_graph = OperationGraph([mul_op])
    simplified_op_graph = simplify(op_graph, ConvertMul(op_graph))

    assert simplified_op_graph.walk(
        EnsureSupportVisitor(
            [
                operations.Input,
                operations.Add,
                operations.MatMul,
                operations.Gemm,
                operations.Conv,
            ]
        )
    )

    x = np.random.randn(*input_op.shape).astype(input_op.dtype)
    y1 = op_graph(x)
    y2 = simplified_op_graph(x)
    assert np.allclose(y1, y2)

    mul_op = operations.Mul(
        np.random.randn(*input_op.shape).astype(input_op.dtype), input_op
    )
    op_graph = OperationGraph([mul_op])
    simplified_op_graph = simplify(op_graph, ConvertMul(op_graph))

    assert simplified_op_graph.walk(
        EnsureSupportVisitor(
            [
                operations.Input,
                operations.Add,
                operations.MatMul,
                operations.Gemm,
                operations.Conv,
            ]
        )
    )

    x = np.random.randn(*input_op.shape).astype(input_op.dtype)
    y1 = op_graph(x)
    y2 = simplified_op_graph(x)
    assert np.allclose(y1, y2)


def test_convert_mul_noop():
    input_op_1 = operations.Input((1,), dtype=np.dtype(np.float64))
    input_op_2 = operations.Input((1,), dtype=np.dtype(np.float64))
    mul_op = operations.Mul(input_op_1, input_op_2)
    op_graph = OperationGraph([mul_op])
    simplified_op_graph = simplify(op_graph, ConvertMul(op_graph))
    assert simplified_op_graph.output_operations[0] == op_graph.output_operations[0]

    input_op = operations.Input((1,), dtype=np.dtype(np.float64))
    mul_op = operations.Mul(input_op, np.random.randn(5).astype(input_op.dtype))
    op_graph = OperationGraph([mul_op])
    simplified_op_graph = simplify(op_graph, ConvertMul(op_graph))
    assert simplified_op_graph.output_operations[0] == op_graph.output_operations[0]

    input_op = operations.Input((), dtype=np.dtype(np.float64))
    mul_op = operations.Mul(input_op, np.random.randn())
    op_graph = OperationGraph([mul_op])
    simplified_op_graph = simplify(op_graph, ConvertMul(op_graph))
    assert simplified_op_graph.output_operations[0] == op_graph.output_operations[0]

    input_op = operations.Input((1, 3, 4, 5), dtype=np.dtype(np.float64))
    mul_op = operations.Mul(input_op, np.random.randn())
    op_graph = OperationGraph([mul_op])
    simplified_op_graph = simplify(op_graph, ConvertMul(op_graph))
    assert simplified_op_graph.output_operations[0] == op_graph.output_operations[0]
