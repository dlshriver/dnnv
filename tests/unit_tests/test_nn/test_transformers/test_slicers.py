import numpy as np
import pytest

from dnnv.nn import OperationGraph, operations
from dnnv.nn.transformers.slicers import Slicer


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


def test_slicer_stop_0(op_graph):
    new_op_graph = Slicer(None, 0).transform(op_graph)
    output_ops = new_op_graph.outputs

    assert len(output_ops) == 0


def test_slicer_all(op_graph):
    new_op_graph = Slicer(None, None).transform(op_graph)
    output_ops = new_op_graph.outputs

    assert len(output_ops) == 1
    output_op = output_ops[0]
    assert isinstance(output_op, operations.Sub)
    assert isinstance(output_op.b, operations.Relu)
    assert isinstance(output_op.b.x, operations.Add)
    assert isinstance(output_op.b.x.a, operations.Mul)
    assert isinstance(output_op.b.x.a.a, operations.Input)
    assert isinstance(output_op.b.x.b, operations.Div)
    assert isinstance(output_op.b.x.b.a, operations.Input)
    assert output_op.b.x.a.a is output_op.b.x.b.a


def test_slicer_end(op_graph):
    new_op_graph = Slicer(-1, None).transform(op_graph)
    output_ops = new_op_graph.outputs

    assert len(output_ops) == 1
    output_op = output_ops[0]
    assert isinstance(output_op, operations.Sub)
    assert isinstance(output_op.b, operations.Input)


def test_slicer_cycle():
    input_op = operations.Input(np.array([1, 5]), np.dtype(np.float32))
    mul_op = operations.Mul(input_op, np.float32(1))
    div_op = operations.Div(input_op, np.float32(2))
    add_op = operations.Add(mul_op, div_op)
    relu_op = operations.Relu(add_op)
    mul_op.b = relu_op
    op_graph = OperationGraph([relu_op])

    with pytest.raises(ValueError, match="Slicing cyclic graphs is not supported."):
        _ = Slicer(None, None).transform(op_graph)
