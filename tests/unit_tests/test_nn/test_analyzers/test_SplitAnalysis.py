import numpy as np

from dnnv.nn.analyzers import SplitAnalysis
from dnnv.nn.graph import OperationGraph
from dnnv.nn.operations import *


def test_empty_op_graph():
    op_graph = OperationGraph([])

    analysis = SplitAnalysis(op_graph)
    assert len(analysis.results) == 0


def test_no_splits():
    input_op = Input((1, 5), np.dtype(np.float32))
    gemm_op = Gemm(input_op, np.random.randn(5, 2), np.zeros(2))
    relu_op = Relu(gemm_op)
    op_graph = OperationGraph([relu_op])

    analysis = SplitAnalysis(op_graph)
    assert analysis[input_op] == False
    assert analysis[gemm_op] == False
    assert analysis[relu_op] == False


def test_with_splits():
    input_op = Input((1, 5), np.dtype(np.float32))
    gemm_op = Gemm(input_op, np.random.randn(5, 2), np.zeros(2))
    relu_op = Relu(gemm_op)
    concat_op = Concat([relu_op, input_op], axis=1)
    op_graph = OperationGraph([concat_op])

    analysis = SplitAnalysis(op_graph)
    assert analysis[input_op] == True
    assert analysis[gemm_op] == False
    assert analysis[relu_op] == False
    assert analysis[concat_op] == False
