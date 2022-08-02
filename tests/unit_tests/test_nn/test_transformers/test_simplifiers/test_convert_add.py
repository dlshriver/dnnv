import numpy as np

from dnnv.nn import OperationGraph, operations
from dnnv.nn.transformers.simplifiers import simplify
from dnnv.nn.transformers.simplifiers.convert_add import ConvertAdd
from dnnv.nn.visitors import EnsureSupportVisitor


def test_convert_add():
    input_op = operations.Input((-1, 5), dtype=np.dtype(np.float64))
    add_op = operations.Add(
        input_op, np.random.randn(1, *input_op.shape[1:]).astype(input_op.dtype)
    )
    op_graph = OperationGraph([add_op])
    simplified_op_graph = simplify(op_graph, ConvertAdd(op_graph))

    assert simplified_op_graph.walk(
        EnsureSupportVisitor([operations.Input, operations.Gemm])
    )

    x = np.random.randn(100, *input_op.shape[1:]).astype(input_op.dtype)
    y1 = op_graph(x)
    y2 = simplified_op_graph(x)
    assert np.allclose(y1, y2)

    add_op = operations.Add(
        np.random.randn(1, *input_op.shape[1:]).astype(input_op.dtype), input_op
    )
    op_graph = OperationGraph([add_op])
    simplified_op_graph = simplify(op_graph, ConvertAdd(op_graph))

    assert simplified_op_graph.walk(
        EnsureSupportVisitor([operations.Input, operations.Gemm])
    )

    x = np.random.randn(100, *input_op.shape[1:]).astype(input_op.dtype)
    y1 = op_graph(x)
    y2 = simplified_op_graph(x)
    assert np.allclose(y1, y2)


def test_convert_add_zero():
    input_op = operations.Input((-1, 3, 4, 5), dtype=np.dtype(np.float64))
    add_op = operations.Add(
        input_op, np.zeros((1, *input_op.shape[1:]), dtype=input_op.dtype)
    )
    op_graph = OperationGraph([add_op])
    simplified_op_graph = simplify(op_graph, ConvertAdd(op_graph))

    assert simplified_op_graph.walk(EnsureSupportVisitor([operations.Input]))

    x = np.random.randn(100, *input_op.shape[1:]).astype(input_op.dtype)
    y1 = op_graph(x)
    y2 = simplified_op_graph(x)
    assert np.allclose(y1, y2)

    add_op = operations.Add(
        np.zeros((1, *input_op.shape[1:]), dtype=input_op.dtype), input_op
    )
    op_graph = OperationGraph([add_op])
    simplified_op_graph = simplify(op_graph, ConvertAdd(op_graph))

    assert simplified_op_graph.walk(EnsureSupportVisitor([operations.Input]))

    x = np.random.randn(100, *input_op.shape[1:]).astype(input_op.dtype)
    y1 = op_graph(x)
    y2 = simplified_op_graph(x)
    assert np.allclose(y1, y2)


def test_convert_add_constants():
    shape = (1, 5)
    dtype = np.float64
    add_op = operations.Add(
        np.random.randn(*shape).astype(dtype), np.random.randn(*shape).astype(dtype)
    )
    op_graph = OperationGraph([add_op])
    simplified_op_graph = simplify(op_graph, ConvertAdd(op_graph))
    assert isinstance(simplified_op_graph.output_operations[0], np.ndarray)
