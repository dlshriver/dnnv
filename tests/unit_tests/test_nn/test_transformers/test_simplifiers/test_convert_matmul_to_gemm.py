import numpy as np

from dnnv.nn import OperationGraph, operations
from dnnv.nn.transformers.simplifiers import simplify
from dnnv.nn.transformers.simplifiers.convert_matmul_to_gemm import ConvertMatMulToGemm
from dnnv.nn.visitors import EnsureSupportVisitor, OperationCounter


def test_convert_matmul():
    input_op = operations.Input((-1, 5), dtype=np.dtype(np.float64))
    matmul_op = operations.MatMul(
        input_op,
        np.random.randn(input_op.shape[1], 10).astype(input_op.dtype),
    )
    op_graph = OperationGraph([matmul_op])
    simplified_op_graph = simplify(op_graph, ConvertMatMulToGemm(op_graph))

    assert simplified_op_graph.walk(
        EnsureSupportVisitor([operations.Input, operations.Gemm])
    )
    assert op_graph.walk(OperationCounter())[-1] == 2
    assert simplified_op_graph.walk(OperationCounter())[-1] == 2

    x = np.random.randn(100, *input_op.shape[1:]).astype(input_op.dtype)
    y1 = op_graph(x)
    y2 = simplified_op_graph(x)
    assert np.allclose(y1, y2)

    input_op = operations.Input((5, -1), dtype=np.dtype(np.float64))
    matmul_op = operations.MatMul(
        np.random.randn(10, input_op.shape[0]).astype(input_op.dtype),
        input_op,
    )
    op_graph = OperationGraph([matmul_op])
    simplified_op_graph = simplify(op_graph, ConvertMatMulToGemm(op_graph))
    simplified_op_graph.pprint()

    assert simplified_op_graph.walk(
        EnsureSupportVisitor([operations.Input, operations.Gemm])
    )
    assert op_graph.walk(OperationCounter())[-1] == 2
    assert simplified_op_graph.walk(OperationCounter())[-1] == 2

    x = np.random.randn(*input_op.shape[:1], 100).astype(input_op.dtype)
    y1 = op_graph(x)
    y2 = simplified_op_graph(x)
    assert np.allclose(y1, y2)
