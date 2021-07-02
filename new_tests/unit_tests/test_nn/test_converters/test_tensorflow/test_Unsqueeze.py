import numpy as np
import pytest

from dnnv.nn.converters.tensorflow import *
from dnnv.nn.operations import *


def test_Unsqueeze():
    x = np.random.randn(1, 3, 1, 5).astype(np.float32)
    axes = np.array([0]).astype(np.int64)
    y = np.expand_dims(x, axis=0)

    op = Unsqueeze(x, axes)
    tf_op = TensorflowConverter().visit(op)
    result = tf_op().numpy()
    assert np.allclose(result, y)

    op = Unsqueeze(Input(x.shape, x.dtype), Input(axes.shape, axes.dtype))
    tf_op = TensorflowConverter().visit(op)
    result = tf_op(x, axes).numpy()
    assert np.allclose(result, y)


def test_Unsqueeze_negative_axes():
    x = np.random.randn(1, 3, 1, 5).astype(np.float32)
    axes = np.array([-2]).astype(np.int64)
    y = np.expand_dims(x, axis=-2)

    op = Unsqueeze(x, axes)
    tf_op = TensorflowConverter().visit(op)
    result = tf_op().numpy()
    assert np.allclose(result, y)


def test_Unsqueeze_one_axis():
    x = np.random.randn(3, 4, 5).astype(np.float32)

    for i in range(x.ndim):
        axes = np.array([i]).astype(np.int64)
        y = np.expand_dims(x, axis=i)

        op = Unsqueeze(x, axes)
        tf_op = TensorflowConverter().visit(op)
        result = tf_op().numpy()
        assert np.allclose(result, y)


def test_Unsqueeze_three_axes():
    x = np.random.randn(3, 4, 5).astype(np.float32)
    axes = np.array([2, 4, 5]).astype(np.int64)
    y = np.expand_dims(x, axis=2)
    y = np.expand_dims(y, axis=4)
    y = np.expand_dims(y, axis=5)

    op = Unsqueeze(x, axes)
    tf_op = TensorflowConverter().visit(op)
    result = tf_op().numpy()
    assert np.allclose(result, y)


def test_Unsqueeze_two_axes():
    x = np.random.randn(3, 4, 5).astype(np.float32)
    axes = np.array([1, 4]).astype(np.int64)
    y = np.expand_dims(x, axis=1)
    y = np.expand_dims(y, axis=4)

    op = Unsqueeze(x, axes)
    tf_op = TensorflowConverter().visit(op)
    result = tf_op().numpy()
    assert np.allclose(result, y)


def test_Unsqueeze_unsorted_axes():
    x = np.random.randn(3, 4, 5).astype(np.float32)
    axes = np.array([5, 4, 2]).astype(np.int64)
    y = np.expand_dims(x, axis=2)
    y = np.expand_dims(y, axis=4)
    y = np.expand_dims(y, axis=5)

    op = Unsqueeze(x, axes)
    tf_op = TensorflowConverter().visit(op)
    result = tf_op().numpy()
    assert np.allclose(result, y)
