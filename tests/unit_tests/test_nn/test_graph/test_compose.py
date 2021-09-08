import numpy as np
import pytest

from dnnv.nn import OperationGraph
from dnnv.nn.operations import *


def test_0():
    op_graph_0 = OperationGraph([Mul(Input((), np.dtype(np.float32)), 2.0)])
    op_graph_1 = OperationGraph([Add(Input((), np.dtype(np.float32)), 2.0)])

    composed_op_graph = op_graph_1.compose(op_graph_0)
    assert len(composed_op_graph.output_operations) == 1
    output_op = composed_op_graph.output_operations[0]
    assert isinstance(output_op, Add)
    assert len(output_op.inputs) == 1
    add_input_op = output_op.inputs[0]
    assert isinstance(add_input_op, Mul)
    assert len(add_input_op.inputs) == 1
    mul_input_op = add_input_op.inputs[0]
    assert isinstance(mul_input_op, Input)


def test_1():
    op_graph_0 = OperationGraph(
        [Input((), np.dtype(np.float32)), Input((), np.dtype(np.float32))]
    )
    op_graph_1 = OperationGraph([Add(Input((), np.dtype(np.float32)), 2.0)])

    with pytest.raises(ValueError) as excinfo:
        composed_op_graph = op_graph_1.compose(op_graph_0)
    assert str(excinfo.value).startswith(
        "Number of inputs and outputs must match for op graph composition."
    )


def test_2():
    op_graph_0 = OperationGraph([Mul(Input((1, 5), np.dtype(np.float32)), 2.0)])
    op_graph_1 = OperationGraph([Add(Input((1, 10), np.dtype(np.float32)), 2.0)])

    with pytest.raises(ValueError) as excinfo:
        composed_op_graph = op_graph_1.compose(op_graph_0)
    assert str(excinfo.value).startswith(
        "Input and output shapes and types must match for op graph composition."
    )


def test_3():
    input_op = Input((1, 5), np.dtype(np.float32))
    mul_op = Mul(input_op, 2.0)
    concat_op = Concat([mul_op, input_op], axis=1)

    input_op_ = Input((1, 10), np.dtype(np.float32))
    add_op = Add(input_op_, 1.0)

    op_graph_0 = OperationGraph([concat_op])
    op_graph_1 = OperationGraph([add_op])

    composed_op_graph = op_graph_1.compose(op_graph_0)

    assert len(composed_op_graph.output_operations) == 1
    output_op = composed_op_graph.output_operations[0]
    assert isinstance(output_op, Add)
    assert len(output_op.inputs) == 1

    add_input_op = output_op.inputs[0]
    assert isinstance(add_input_op, Concat)
    assert len(add_input_op.inputs) == 2

    concat_input_op_0 = add_input_op.inputs[0]
    concat_input_op_1 = add_input_op.inputs[1]
    assert isinstance(concat_input_op_0, Mul)
    assert isinstance(concat_input_op_1, Input)

    assert len(concat_input_op_0.inputs) == 1
    mul_input_op = concat_input_op_0.inputs[0]
    assert isinstance(mul_input_op, Input)


def test_4():
    input_op = Input((1, 5), np.dtype(np.float32))
    add_op = Add(input_op, 1.0)

    input_op_ = Input((1, 5), np.dtype(np.float32))
    mul_op = Mul(input_op_, 2.0)
    concat_op = Concat([mul_op, input_op_], axis=1)

    op_graph_0 = OperationGraph([add_op])
    op_graph_1 = OperationGraph([concat_op])

    composed_op_graph = op_graph_1.compose(op_graph_0)

    composed_op_graph.pprint()

    assert len(composed_op_graph.output_operations) == 1
    output_op = composed_op_graph.output_operations[0]
    assert isinstance(output_op, Concat)
    assert len(output_op.inputs) == 2

    concat_input_op_0 = output_op.inputs[0]
    concat_input_op_1 = output_op.inputs[1]
    assert isinstance(concat_input_op_0, Mul)
    assert isinstance(concat_input_op_1, Add)

    concat_0_input_op = concat_input_op_0.inputs[0]
    assert isinstance(concat_0_input_op, Add)
    assert len(concat_0_input_op.inputs) == 1

    add_input_op = concat_0_input_op.inputs[0]
    assert isinstance(add_input_op, Input)
    assert len(add_input_op.inputs) == 0

    concat_1_input_op = concat_input_op_1.inputs[0]
    assert isinstance(concat_1_input_op, Input)
    assert len(concat_1_input_op.inputs) == 0
