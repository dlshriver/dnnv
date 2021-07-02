import numpy as np
import pytest

from dnnv.nn import OperationGraph
from dnnv.nn.operations import *


def test_simple_linear_no_bounds(capsys):
    input_op = Input((1, 5), np.dtype(np.float32))
    mul_op_0 = Mul(input_op, 2.0)
    add_op_0 = Add(mul_op_0, 1.0)
    relu_op_0 = Relu(add_op_0)
    mul_op_1 = Mul(relu_op_0, 2.0)
    add_op_1 = Add(mul_op_1, 1.0)

    op_graph = OperationGraph([add_op_1])
    _ = capsys.readouterr()
    op_graph.pprint()
    captured_0 = capsys.readouterr()

    _op_graph = op_graph[:]
    _ = capsys.readouterr()
    _op_graph.pprint()
    captured_1 = capsys.readouterr()

    assert captured_0 == captured_1


def test_split_paths_no_bounds(capsys):
    input_op = Input((1, 5), np.dtype(np.float32))
    mul_op = Mul(input_op, 2.0)
    concat_op = Concat([mul_op, input_op], axis=1)
    add_op = Add(concat_op, 1.0)

    op_graph = OperationGraph([add_op])
    _ = capsys.readouterr()
    op_graph.pprint()
    captured_0 = capsys.readouterr()

    _op_graph = op_graph[:]
    _ = capsys.readouterr()
    _op_graph.pprint()
    captured_1 = capsys.readouterr()

    assert captured_0 == captured_1


def test_simple_linear_select_output(capsys):
    input_op = Input((1, 5), np.dtype(np.float32))
    mul_op_0 = Mul(input_op, 2.0)
    add_op_0 = Add(mul_op_0, 1.0)
    relu_op_0 = Relu(add_op_0)
    mul_op_1 = Mul(relu_op_0, 2.0)
    add_op_1 = Add(mul_op_1, 1.0)

    op_graph = OperationGraph([add_op_1])
    _ = capsys.readouterr()
    op_graph.pprint()
    captured_0 = capsys.readouterr()

    _op_graph = op_graph[0]
    _ = capsys.readouterr()
    _op_graph.pprint()
    captured_1 = capsys.readouterr()

    assert captured_0 == captured_1


def test_split_paths_select_output(capsys):
    input_op = Input((1, 5), np.dtype(np.float32))
    mul_op = Mul(input_op, 2.0)
    concat_op = Concat([mul_op, input_op], axis=1)
    add_op = Add(concat_op, 1.0)

    op_graph = OperationGraph([add_op])
    _ = capsys.readouterr()
    op_graph.pprint()
    captured_0 = capsys.readouterr()

    _op_graph = op_graph[0]
    _ = capsys.readouterr()
    _op_graph.pprint()
    captured_1 = capsys.readouterr()

    assert captured_0 == captured_1


def test_tuple_index_type_errors():
    input_op = Input((1, 5), np.dtype(np.float32))
    mul_op_0 = Mul(input_op, 2.0)
    add_op_0 = Add(mul_op_0, 1.0)
    relu_op_0 = Relu(add_op_0)
    mul_op_1 = Mul(relu_op_0, 2.0)
    add_op_1 = Add(mul_op_1, 1.0)

    op_graph = OperationGraph([add_op_1])

    with pytest.raises(TypeError) as excinfo:
        _op_graph = op_graph[...]
    assert str(excinfo.value).startswith(
        "Unsupported type for indexing operation graph: "
    )

    with pytest.raises(TypeError) as excinfo:
        _op_graph = op_graph[1, 2, 3, 4, 5]
    assert str(excinfo.value).startswith("Unsupported indexing expression")

    with pytest.raises(TypeError) as excinfo:
        _op_graph = op_graph[0, 0]
    assert str(excinfo.value).startswith("Unsupported type for slicing indices: ")

    with pytest.raises(TypeError) as excinfo:
        _op_graph = op_graph[:, :]
    assert str(excinfo.value).startswith("Unsupported type for selecting operations: ")


def test_tuple_index_value_errors():
    input_op = Input((1, 5), np.dtype(np.float32))
    mul_op_0 = Mul(input_op, 2.0)
    add_op_0 = Add(mul_op_0, 1.0)
    relu_op_0 = Relu(add_op_0)
    mul_op_1 = Mul(relu_op_0, 2.0)
    add_op_1 = Add(mul_op_1, 1.0)

    op_graph = OperationGraph([add_op_1])

    with pytest.raises(ValueError) as excinfo:
        _op_graph = op_graph[::2, 0]
    assert str(excinfo.value).startswith("Slicing does not support non-unit steps.")
