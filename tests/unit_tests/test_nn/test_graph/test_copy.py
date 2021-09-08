import numpy as np

from dnnv.nn import OperationGraph
from dnnv.nn.operations import *


def test_0(capsys):
    input_op_ = Input((1, 5), np.dtype(np.float32))
    mul_op = Mul(input_op_, 2.0)
    concat_op = Concat([mul_op, input_op_], axis=1)

    op_graph = OperationGraph([concat_op])
    _ = capsys.readouterr()
    op_graph.pprint()
    captured_0 = capsys.readouterr()

    op_graph_copy = op_graph.copy()
    _ = capsys.readouterr()
    op_graph_copy.pprint()
    captured_1 = capsys.readouterr()

    assert captured_0 == captured_1
