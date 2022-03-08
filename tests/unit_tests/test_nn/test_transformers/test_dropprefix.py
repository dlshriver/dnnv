import numpy as np
import pytest

from dnnv.nn import OperationGraph, operations
from dnnv.nn.transformers.dropprefix import DropPrefix


@pytest.fixture
def op_graph():
    input_op = operations.Input(np.array([1, 5]), np.dtype(np.float32))
    mul_op = operations.Mul(input_op, np.float32(1))
    div_op = operations.Div(input_op, np.float32(2))
    add_op = operations.Add(mul_op, div_op)
    relu_op = operations.Relu(add_op)
    sub_op = operations.Sub(np.zeros((1, 5), dtype=np.float32), relu_op)
    op_graph = OperationGraph([sub_op])
    return op_graph


def test_drop_prefix(op_graph):
    prefix = op_graph[:3]
    new_op_graph = DropPrefix(prefix).transform(op_graph)

    assert len(new_op_graph.outputs) == 1
    output_op = new_op_graph.outputs[0]
    assert isinstance(output_op, operations.Sub)
    assert isinstance(output_op.b, operations.Relu)
    assert isinstance(output_op.b.x, operations.Input)
