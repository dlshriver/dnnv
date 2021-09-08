import numpy as np

from dnnv.nn import OperationGraph
from dnnv.nn.operations import *


def test_0(capsys):
    op_graph = OperationGraph([Input((1, 2, 3, 4), np.dtype(np.float64))])
    op_graph.pprint()
    captured = capsys.readouterr()
    expected_output = """\
Input_0                         : Input((1, 2, 3, 4), dtype=float64)
"""
    assert captured.out == expected_output


def test_1(capsys):
    op_graph = OperationGraph([Add(Input((1,), np.dtype(np.float32)), np.float32(6))])
    op_graph.pprint()
    captured = capsys.readouterr()
    expected_output = """\
Input_0                         : Input((1,), dtype=float32)
Add_0                           : Add(Input_0, float32_0)
"""
    assert captured.out == expected_output


def test_2(capsys):
    input_op = Input((1, 5), np.dtype(np.float32))
    mul_op = Mul(input_op, np.float32(1))
    div_op = Div(input_op, np.float32(2))
    add_op = Add(mul_op, div_op)
    op_graph = OperationGraph([add_op])
    op_graph.pprint()
    captured = capsys.readouterr()
    expected_output = """\
Input_0                         : Input((1, 5), dtype=float32)
Mul_0                           : Mul(Input_0, float32_0)
Div_0                           : Div(Input_0, float32_1)
Add_0                           : Add(Mul_0, Div_0)
"""
    assert captured.out == expected_output
