import numpy as np

from dnnv.nn.analyzers import ConstantsAnalysis
from dnnv.nn.graph import OperationGraph
from dnnv.nn.operations import *


def test_empty_op_graph():
    op_graph = OperationGraph([])

    analysis = ConstantsAnalysis(op_graph)
    assert len(analysis.results) == 0


def test_no_constants():
    input_op = Input((1, 5), np.dtype(np.float32))
    gemm_op = Gemm(input_op, np.random.randn(5, 2), np.zeros(2))
    relu_op = Relu(gemm_op)
    op_graph = OperationGraph([relu_op])

    analysis = ConstantsAnalysis(op_graph)
    assert analysis[input_op] == False
    assert analysis[gemm_op] == False
    assert analysis[relu_op] == False


def test_with_constant_ops():
    input_op = Input(np.array((1, 3, 4, 4)), np.dtype(np.float32))
    shape_op = Shape(input_op)
    gather_op = Gather(shape_op, np.array(0))
    unsqueeze_op = Unsqueeze(gather_op, np.array([0]))
    concat_op = Concat([unsqueeze_op, np.array([-1])], 0)
    reshape_op = Reshape(input_op, concat_op)
    op_graph = OperationGraph([reshape_op])

    analysis = ConstantsAnalysis(op_graph)
    assert analysis[input_op] == False
    assert analysis[shape_op] == True
    assert analysis[gather_op] == True
    assert analysis[unsqueeze_op] == True
    assert analysis[concat_op] == True
    assert analysis[reshape_op] == False
