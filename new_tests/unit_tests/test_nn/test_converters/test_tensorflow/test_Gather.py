import numpy as np
import pytest

from dnnv.nn.converters.tensorflow import *
from dnnv.nn.operations import *


def test_Gather_0():
    data = np.random.randn(5, 4, 3, 2).astype(np.float32)
    indices = np.array([0, 1, 3])
    op = Gather(data, indices, axis=0)
    tf_op = TensorflowConverter().visit(op)
    result = tf_op().numpy()
    y = np.take(data, indices, axis=0)
    assert result.shape == y.shape
    assert np.allclose(result, y)

    data = np.random.randn(5, 4, 3, 2).astype(np.float32)
    indices = np.array([0, 1, 3])
    data_input_op = Input((5, 4, 3, 2), np.dtype(np.float32))
    op = Gather(data_input_op, indices, axis=0)
    tf_op = TensorflowConverter().visit(op)
    result = tf_op(data).numpy()
    y = np.take(data, indices, axis=0)
    assert result.shape == y.shape
    assert np.allclose(result, y)

    data = np.random.randn(5, 4, 3, 2).astype(np.float32)
    indices = np.array([0, 1, 3])
    indices_input_op = Input((3,), np.dtype(np.int64))
    op = Gather(data, indices_input_op, axis=0)
    tf_op = TensorflowConverter().visit(op)
    result = tf_op(indices).numpy()
    y = np.take(data, indices, axis=0)
    assert result.shape == y.shape
    assert np.allclose(result, y)


def test_Gather_1():
    data = np.random.randn(5, 4, 3, 2).astype(np.float32)
    indices = np.array([0, 1, 3])
    op = Gather(data, indices, axis=1)
    tf_op = TensorflowConverter().visit(op)
    result = tf_op().numpy()
    y = np.take(data, indices, axis=1)
    assert result.shape == y.shape
    assert np.allclose(result, y)


def test_Gather_2d_indices():
    data = np.random.randn(3, 3).astype(np.float32)
    indices = np.array([[0, 2]])
    op = Gather(data, indices, axis=1)
    tf_op = TensorflowConverter().visit(op)
    result = tf_op().numpy()
    y = np.take(data, indices, axis=1)
    assert result.shape == y.shape
    assert np.allclose(result, y)


@pytest.mark.xfail
def test_Gather_negative_indices():
    data = np.arange(10).astype(np.float32)
    indices = np.array([0, -9, -10])
    op = Gather(data, indices, axis=0)
    tf_op = TensorflowConverter().visit(op)
    result = tf_op().numpy()
    y = np.take(data, indices, axis=0)
    assert result.shape == y.shape
    assert np.allclose(result, y)
