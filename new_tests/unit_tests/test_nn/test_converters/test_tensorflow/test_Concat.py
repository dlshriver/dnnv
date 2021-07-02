import numpy as np

from dnnv.nn.converters.tensorflow import *
from dnnv.nn.operations import *


def test_Concat_consts():
    x0 = np.arange(5)
    x1 = np.arange(10, 20)

    op = Concat([x0, x1], 0)
    tf_op = TensorflowConverter().visit(op)
    result = tf_op().numpy()
    y = np.array([0, 1, 2, 3, 4, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19])
    assert np.all(result == y)


def test_Concat_x0_is_op():
    x0 = np.arange(5)
    x1 = np.arange(10, 20)

    input_op0 = Input((5,), np.dtype(np.int64))
    op = Concat([input_op0, x1], 0)
    tf_op = TensorflowConverter().visit(op)
    result = tf_op(x0).numpy()
    y = np.array([0, 1, 2, 3, 4, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19])
    assert np.all(result == y)


def test_Concat_x1_is_op():
    x0 = np.arange(5)
    x1 = np.arange(10, 20)

    input_op1 = Input((10,), np.dtype(np.int64))
    op = Concat([x0, input_op1], 0)
    tf_op = TensorflowConverter().visit(op)
    result = tf_op(x1).numpy()
    y = np.array([0, 1, 2, 3, 4, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19])
    assert np.all(result == y)


def test_Concat_x0_x1_are_op():
    x0 = np.arange(5)
    x1 = np.arange(10, 20)

    input_op0 = Input((5,), np.dtype(np.int64))
    input_op1 = Input((10,), np.dtype(np.int64))
    op = Concat([input_op0, input_op1], 0)
    tf_op = TensorflowConverter().visit(op)
    result = tf_op(x0, x1).numpy()
    y = np.array([0, 1, 2, 3, 4, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19])
    assert np.all(result == y)


def test_Concat_1d():
    x0 = np.array([1, 2], dtype=np.float32)
    x1 = np.array([3, 4], dtype=np.float32)

    op = Concat([x0, x1], 0)
    tf_op = TensorflowConverter().visit(op)
    result = tf_op().numpy()
    y = np.array([1, 2, 3, 4], dtype=np.float32)
    assert np.all(result == y)

    op = Concat([x0, x1], -1)
    tf_op = TensorflowConverter().visit(op)
    result = tf_op().numpy()
    y = np.array([1, 2, 3, 4], dtype=np.float32)
    assert np.all(result == y)


def test_Concat_2d():
    x0 = np.array([[1, 2], [3, 4]], dtype=np.float32)
    x1 = np.array([[5, 6], [7, 8]], dtype=np.float32)

    op = Concat([x0, x1], 0)
    tf_op = TensorflowConverter().visit(op)
    result = tf_op().numpy()
    y = np.array([[1, 2], [3, 4], [5, 6], [7, 8]], dtype=np.float32)
    assert np.all(result == y)

    op = Concat([x0, x1], 1)
    tf_op = TensorflowConverter().visit(op)
    result = tf_op().numpy()
    y = np.array([[1, 2, 5, 6], [3, 4, 7, 8]], dtype=np.float32)
    assert np.all(result == y)

    op = Concat([x0, x1], -1)
    tf_op = TensorflowConverter().visit(op)
    result = tf_op().numpy()
    y = np.array([[1, 2, 5, 6], [3, 4, 7, 8]], dtype=np.float32)
    assert np.all(result == y)

    op = Concat([x0, x1], -2)
    tf_op = TensorflowConverter().visit(op)
    result = tf_op().numpy()
    y = np.array([[1, 2], [3, 4], [5, 6], [7, 8]], dtype=np.float32)
    assert np.all(result == y)


def test_Concat_3d():
    x0 = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.float32)
    x1 = np.array([[[9, 10], [11, 12]], [[13, 14], [15, 16]]], dtype=np.float32)

    op = Concat([x0, x1], 0)
    tf_op = TensorflowConverter().visit(op)
    result = tf_op().numpy()
    y = np.array(
        [[[1, 2], [3, 4]], [[5, 6], [7, 8]], [[9, 10], [11, 12]], [[13, 14], [15, 16]]],
        dtype=np.float32,
    )
    assert np.all(result == y)

    op = Concat([x0, x1], 1)
    tf_op = TensorflowConverter().visit(op)
    result = tf_op().numpy()
    y = np.array(
        [[[1, 2], [3, 4], [9, 10], [11, 12]], [[5, 6], [7, 8], [13, 14], [15, 16]]],
        dtype=np.float32,
    )
    assert np.all(result == y)

    op = Concat([x0, x1], 2)
    tf_op = TensorflowConverter().visit(op)
    result = tf_op().numpy()
    y = np.array(
        [[[1, 2, 9, 10], [3, 4, 11, 12]], [[5, 6, 13, 14], [7, 8, 15, 16]]],
        dtype=np.float32,
    )
    assert np.all(result == y)

    op = Concat([x0, x1], -1)
    tf_op = TensorflowConverter().visit(op)
    result = tf_op().numpy()
    y = np.array(
        [[[1, 2, 9, 10], [3, 4, 11, 12]], [[5, 6, 13, 14], [7, 8, 15, 16]]],
        dtype=np.float32,
    )
    assert np.all(result == y)

    op = Concat([x0, x1], -2)
    tf_op = TensorflowConverter().visit(op)
    result = tf_op().numpy()
    y = np.array(
        [[[1, 2], [3, 4], [9, 10], [11, 12]], [[5, 6], [7, 8], [13, 14], [15, 16]]],
        dtype=np.float32,
    )
    assert np.all(result == y)

    op = Concat([x0, x1], -3)
    tf_op = TensorflowConverter().visit(op)
    result = tf_op().numpy()
    y = np.array(
        [[[1, 2], [3, 4]], [[5, 6], [7, 8]], [[9, 10], [11, 12]], [[13, 14], [15, 16]]],
        dtype=np.float32,
    )
    assert np.all(result == y)
