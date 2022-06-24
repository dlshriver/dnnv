import numpy as np

from dnnv.nn import OperationGraph, operations
from dnnv.nn.transformers.simplifiers import simplify
from dnnv.nn.transformers.simplifiers.convert_div_to_mul import ConvertDivToMul
from dnnv.nn.visitors import EnsureSupportVisitor, OperationCounter


def test_convert_div_by_ones():
    input_op = operations.Input((-1, 5), dtype=np.dtype(np.float64))
    div_op = operations.Div(
        input_op,
        np.ones(*input_op.shape[1:], dtype=input_op.dtype),
    )
    op_graph = OperationGraph([div_op])
    simplified_op_graph = simplify(op_graph, ConvertDivToMul(op_graph))

    assert simplified_op_graph.walk(EnsureSupportVisitor([operations.Input]))
    assert op_graph.walk(OperationCounter())[-1] == 2
    assert simplified_op_graph.walk(OperationCounter())[-1] == 1

    x = np.random.randn(100, *input_op.shape[1:]).astype(input_op.dtype)
    y1 = op_graph(x)
    y2 = simplified_op_graph(x)
    assert np.allclose(y1, y2)


def test_convert_div_zero():
    input_op = operations.Input((-1, 5), dtype=np.dtype(np.float64))
    div_op = operations.Div(
        np.zeros(*input_op.shape[1:], dtype=input_op.dtype),
        input_op,
    )
    op_graph = OperationGraph([div_op])
    simplified_op_graph = simplify(op_graph, ConvertDivToMul(op_graph))
    assert isinstance(simplified_op_graph.output_operations[0], np.ndarray)
    x = np.random.randn(1, *input_op.shape[1:]).astype(input_op.dtype)
    y1 = op_graph(x)
    y2 = simplified_op_graph.output_operations[0]
    assert np.allclose(y1, y2)
