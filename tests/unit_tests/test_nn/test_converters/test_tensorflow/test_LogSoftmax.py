import numpy as np
import pytest

from dnnv.nn.converters.tensorflow import *
from dnnv.nn.operations import *


def test_LogSoftmax():
    x = np.array([[-1, 0, 1]]).astype(np.float32)
    y = np.array([[-2.4076061, -1.407606, -0.407606]])

    op = LogSoftmax(x)
    tf_op = TensorflowConverter().visit(op)
    result = tf_op().numpy()
    assert np.allclose(result, y)

    op = LogSoftmax(Input((1, 3), np.dtype(np.float32)))
    tf_op = TensorflowConverter().visit(op)
    result = tf_op(x).numpy()
    assert np.allclose(result, y)


def test_LogSoftmax_axis_default():
    x = np.array([[0, 1, 2, 3], [10000, 10001, 10002, 10003]]).astype(np.float32)
    y = np.array(
        [
            [-3.4401896, -2.4401896, -1.4401896, -0.44018966],
            [-3.4401896, -2.4401896, -1.4401896, -0.44018966],
        ]
    )

    op = LogSoftmax(x)
    tf_op = TensorflowConverter().visit(op)
    result = tf_op().numpy()
    assert np.allclose(result, y)


def test_LogSoftmax_axis_0():
    axis = 0
    x = np.abs(np.random.randn(3, 4, 5).astype(np.float32))
    x_max = np.max(x, axis=axis, keepdims=True)
    tmp = np.exp(x - x_max)
    s = np.sum(tmp, axis=axis, keepdims=True)
    y = (x - x_max) - np.log(s)

    op = LogSoftmax(x, axis=axis)
    tf_op = TensorflowConverter().visit(op)
    result = tf_op().numpy()
    assert np.allclose(result, y)


def test_LogSoftmax_axis_1():
    axis = 1
    x = np.abs(np.random.randn(3, 4, 5).astype(np.float32))
    x_max = np.max(x, axis=axis, keepdims=True)
    tmp = np.exp(x - x_max)
    s = np.sum(tmp, axis=axis, keepdims=True)
    y = (x - x_max) - np.log(s)

    op = LogSoftmax(x, axis=axis)
    tf_op = TensorflowConverter().visit(op)
    result = tf_op().numpy()
    assert np.allclose(result, y)


def test_LogSoftmax_axis_2():
    axis = 2
    x = np.abs(np.random.randn(3, 4, 5).astype(np.float32))
    x_max = np.max(x, axis=axis, keepdims=True)
    tmp = np.exp(x - x_max)
    s = np.sum(tmp, axis=axis, keepdims=True)
    y = (x - x_max) - np.log(s)

    op = LogSoftmax(x, axis=axis)
    tf_op = TensorflowConverter().visit(op)
    result = tf_op().numpy()
    assert np.allclose(result, y)


def test_LogSoftmax_axis_neg1():
    axis = -1
    x = np.abs(np.random.randn(3, 4, 5).astype(np.float32))
    x_max = np.max(x, axis=axis, keepdims=True)
    tmp = np.exp(x - x_max)
    s = np.sum(tmp, axis=axis, keepdims=True)
    y = (x - x_max) - np.log(s)

    op = LogSoftmax(x, axis=axis)
    tf_op = TensorflowConverter().visit(op)
    result = tf_op().numpy()
    assert np.allclose(result, y)
