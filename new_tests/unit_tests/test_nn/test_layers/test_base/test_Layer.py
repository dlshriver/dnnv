import numpy as np
import pytest

from dnnv.nn.graph import OperationGraph
from dnnv.nn.layers import Layer, Convolutional
from dnnv.nn.operations import *


def test_Layer():
    layer = Layer()
    assert isinstance(layer, Layer)


def test_new_layer_no_OP_PATTERN():
    with pytest.raises(TypeError) as excinfo:

        class TestLayer(Layer):
            pass

    assert str(excinfo.value) == "Layer TestLayer must specify `OP_PATTERN`"


def test_new_layer_bad_OP_PATTERN():
    with pytest.raises(TypeError) as excinfo:

        class TestLayer(Layer):
            OP_PATTERN = "bad value"

    assert str(excinfo.value) == "`OP_PATTERN` must be an operation pattern"


def test_match_with_layer_types_fail():
    op = Input((-1, 3, 2, 2), np.dtype(np.float32))
    W = np.ones((3, 5, 2, 2), dtype=np.float32)
    b = np.zeros(2, dtype=np.float32)
    conv_op = Conv(op, W, b)
    op_graph = OperationGraph([conv_op])
    with pytest.raises(TypeError) as excinfo:
        layer_match = Convolutional.match(op_graph, layer_types=[Convolutional])
    assert (
        str(excinfo.value) == "match() got an unexpected keyword argument 'layer_types'"
    )


def test_match_provided_layer_types():
    op = Input((-1, 3, 2, 2), np.dtype(np.float32))
    W = np.ones((3, 5, 2, 2), dtype=np.float32)
    b = np.zeros(2, dtype=np.float32)
    conv_op = Conv(op, W, b)
    op_graph = OperationGraph([conv_op])
    layer_match = Layer.match(op_graph, layer_types=[Convolutional])
    assert layer_match is not None
    assert isinstance(layer_match.layer, Convolutional)


def test_match_convolutional():
    op = Input((-1, 3, 2, 2), np.dtype(np.float32))
    W = np.ones((3, 5, 2, 2), dtype=np.float32)
    b = np.zeros(2, dtype=np.float32)
    conv_op = Conv(op, W, b)
    op_graph = OperationGraph([conv_op])
    layer_match = Layer.match(op_graph)
    assert layer_match is not None
    assert isinstance(layer_match.layer, Convolutional)


def test_match_op_pattern_none():
    class FakeLayer(Layer):
        OP_PATTERN = None

    op = Input((-1, 3, 2, 2), np.dtype(np.float32))
    atan_op = Atan(op)
    op_graph = OperationGraph([atan_op])
    layer_match = Layer.match(op_graph)
    # TODO : is this right? or should None match anything?
    assert layer_match is None
