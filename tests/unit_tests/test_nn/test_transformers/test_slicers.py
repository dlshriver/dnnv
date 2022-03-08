import numpy as np
import pytest

from dnnv.nn import OperationGraph, operations
from dnnv.nn.transformers.slicers import DropPrefix, Slicer


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
    output_ops = Slicer(None, 0).walk(op_graph)[0]

    assert len(output_ops) == 0


def test_slicer_all(op_graph):
    output_ops = Slicer(None, None).walk(op_graph)[0]

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
    output_ops = Slicer(-1, None).walk(op_graph)[0]

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
        output_ops = Slicer(None, None).walk(op_graph)[0]


def test_drop_prefix(op_graph):
    prefix = op_graph[:3]
    output_ops = DropPrefix(prefix).transform(op_graph)

    assert len(output_ops) == 1
    output_op = output_ops[0]
    assert isinstance(output_op, operations.Sub)
    assert isinstance(output_op.b, operations.Relu)
    assert isinstance(output_op.b.x, operations.Input)
