import numpy as np

from dnnv.nn import OperationGraph
from dnnv.nn.operations import *


def test_0():
    op_graph = OperationGraph([Input((1, 2, 3, 4), np.dtype(np.float64))])
    output_details = op_graph.output_details
    assert len(output_details) == 1
    assert output_details[0].shape == (1, 2, 3, 4)
    assert output_details[0].dtype == np.dtype(np.float64)
    assert op_graph._output_details is output_details
    _output_details = output_details
    output_details = op_graph.output_details
    assert _output_details is output_details
    assert len(output_details) == 1
    assert output_details[0].shape == (1, 2, 3, 4)
    assert output_details[0].dtype == np.dtype(np.float64)


def test_1():
    op_graph = OperationGraph([Add(Input((1,), np.dtype(np.float32)), np.float32(6))])
    output_details = op_graph.output_details
    assert len(output_details) == 1
    assert output_details[0].shape == (1,)
    assert output_details[0].dtype == np.dtype(np.float32)


def test_2():
    input_op = Input((1, 5), np.dtype(np.float32))
    mul_op = Mul(input_op, np.float32(1))
    div_op = Div(input_op, np.float32(2))
    add_op = Add(mul_op, div_op)
    op_graph = OperationGraph([add_op])
    output_details = op_graph.output_details
    assert len(output_details) == 1
    assert output_details[0].shape == (1, 5)
    assert output_details[0].dtype == np.dtype(np.float32)


def test_3():
    input_op_0 = Input((1, 5), np.dtype(np.float32))
    input_op_1 = Input((1, 5), np.dtype(np.float32))
    add_op = Add(input_op_0, input_op_1)
    op_graph = OperationGraph([add_op])
    output_details = op_graph.output_details
    assert len(output_details) == 1
    assert output_details[0].shape == (1, 5)
    assert output_details[0].dtype == np.dtype(np.float32)


def test_4():
    input_op_0 = Input((1, 5, 2, 2), np.dtype(np.float32))
    input_op_1 = Input((1, 20), np.dtype(np.float32))
    flatten_op = Flatten(input_op_0, axis=1)
    add_op = Add(flatten_op, input_op_1)
    op_graph = OperationGraph([add_op])
    output_details = op_graph.output_details
    assert len(output_details) == 1
    assert output_details[0].shape == (1, 20)
    assert output_details[0].dtype == np.dtype(np.float32)


def test_5():
    input_op = Input((1, 5), np.dtype(np.float32))
    gemm_op_0 = Gemm(
        input_op, np.ones((5, 10), dtype=np.float32), np.zeros((10,), dtype=np.float32)
    )
    gemm_op_1 = Gemm(
        input_op, np.eye(5, dtype=np.float32), np.ones((5,), dtype=np.float32)
    )
    op_graph = OperationGraph([gemm_op_0, gemm_op_1])
    output_details = op_graph.output_details
    assert len(output_details) == 2
    assert output_details[0].shape == (1, 10)
    assert output_details[0].dtype == np.dtype(np.float32)
    assert output_details[1].shape == (1, 5)
    assert output_details[1].dtype == np.dtype(np.float32)


def test_6():
    input_op_0 = Input((1, 1, 2, 2), np.dtype(np.float32))
    input_op_1 = Input((1, 4), np.dtype(np.float32))
    flatten_op = Flatten(input_op_0, axis=1)
    add_op = Add(flatten_op, input_op_1)
    sub_op = Sub(input_op_1, flatten_op)
    op_graph = OperationGraph([sub_op, add_op])
    output_details = op_graph.output_details
    assert len(output_details) == 2
    assert output_details[0].shape == (1, 4)
    assert output_details[0].dtype == np.dtype(np.float32)
    assert output_details[1].shape == (1, 4)
    assert output_details[1].dtype == np.dtype(np.float32)
