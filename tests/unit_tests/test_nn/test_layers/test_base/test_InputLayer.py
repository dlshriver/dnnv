import numpy as np

from dnnv.nn.graph import OperationGraph
from dnnv.nn.layers import InputLayer
from dnnv.nn.operations import *


def test_from_operation_graph():
    op = Input((-1, 5), np.dtype(np.float32))
    op_graph = OperationGraph([op])
    layer = InputLayer.from_operation_graph(op_graph)
    assert isinstance(layer, InputLayer)


def test_match():
    op = Input((-1, 5), np.dtype(np.float32))
    op_graph = OperationGraph([op])
    layer_match = InputLayer.match(op_graph)
    assert layer_match is not None
    assert isinstance(layer_match.layer, InputLayer)
    assert len(layer_match.input_op_graph.output_operations) == 0


def test_match_false():
    op = Input((-1, 5), np.dtype(np.float32))
    add_op = Add(op, 1.0)
    op_graph = OperationGraph([add_op])
    layer_match = InputLayer.match(op_graph)
    assert layer_match is None
