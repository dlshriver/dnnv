import numpy as np
import pytest

from dnnv.nn.converters.tensorflow import *
from dnnv.nn.operations import *


def test_Pad():
    x = np.random.randn(1, 3, 4, 5).astype(np.float32)
    pads = np.array([0, 0, 1, 3, 0, 0, 2, 4]).astype(np.int64)
    value = np.float32(1.2)
    y = np.full((1, 3, 7, 12), value)
    y[:, :, 1:-2, 3:-4] = x

    op = Pad(x, pads, value=value)
    tf_op = TensorflowConverter().visit(op)
    result = tf_op().numpy()
    assert np.allclose(result, y)

    op = Pad(Input((1, 3, 4, 5), np.dtype(np.float32)), pads, value=value)
    tf_op = TensorflowConverter().visit(op)
    result = tf_op(x).numpy()
    assert np.allclose(result, y)


def test_Pad_edge():
    x = np.random.randn(1, 3, 4, 5).astype(np.int32)
    pads = np.array([0, 0, 1, 1, 0, 0, 1, 1]).astype(np.int64)

    op = Pad(x, pads, mode="edge")
    tf_op = TensorflowConverter().visit(op)
    with pytest.raises(TensorflowConverterError):
        result = tf_op().numpy()
    # assert np.allclose(result, y)


def test_Pad_reflect():
    x = np.random.randn(1, 3, 4, 5).astype(np.int32)
    pads = np.array([0, 0, 1, 1, 0, 0, 1, 1]).astype(np.int64)

    op = Pad(x, pads, mode="reflect")
    tf_op = TensorflowConverter().visit(op)
    with pytest.raises(TensorflowConverterError):
        result = tf_op().numpy()
    # assert np.allclose(result, y)
