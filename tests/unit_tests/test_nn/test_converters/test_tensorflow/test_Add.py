import numpy as np

from dnnv.nn.converters.tensorflow import *
from dnnv.nn.operations import *


def test_Add_consts():
    op = Add(3, 4)
    tf_op = TensorflowConverter().visit(op)
    result = tf_op().numpy()
    assert result == 7


def test_Add_a_is_op():
    op = Add(Input((-1, 5), np.dtype(np.float32)), 1)
    tf_op = TensorflowConverter().visit(op)
    x = np.array([[1.0, 2.0, 3.0, 4.0, 5.0]], dtype=np.float32)
    result = tf_op(x).numpy()
    y = np.array([[2.0, 3.0, 4.0, 5.0, 6.0]], dtype=np.float32)
    assert np.all(result == y)


def test_Add_b_is_op():
    op = Add(0.5, Input((-1, 5), np.dtype(np.float32)))
    tf_op = TensorflowConverter().visit(op)
    x = np.array([[-1.0, -2.0, -3.0, -4.0, -5.0]], dtype=np.float32)
    result = tf_op(x).numpy()
    y = np.array([[-0.5, -1.5, -2.5, -3.5, -4.5]], dtype=np.float32)
    assert np.all(result == y)


def test_Add_a_b_are_ops():
    op = Add(Input((-1, 5), np.dtype(np.float32)), Input((-1, 5), np.dtype(np.float32)))
    tf_op = TensorflowConverter().visit(op)
    a = np.array([[-1.0, -2.0, -3.0, -4.0, -5.0]], dtype=np.float32)
    b = np.array([[1.0, 2.0, 3.0, 4.0, 5.0]], dtype=np.float32)
    result = tf_op(a, b).numpy()
    y = np.zeros((1, 5), dtype=np.float32)
    assert np.all(result == y)
