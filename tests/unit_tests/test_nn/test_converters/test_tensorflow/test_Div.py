import numpy as np

from dnnv.nn.converters.tensorflow import *
from dnnv.nn.operations import *

TOL = 1e-6


def test_Div_consts():
    a = np.array([3, 4]).astype(np.float32)
    b = np.array([1, 2]).astype(np.float32)
    op = Div(a, b)
    tf_op = TensorflowConverter().visit(op)
    result = tf_op().numpy()
    y = np.array([3, 2]).astype(np.float32)
    assert np.all(result >= (y - TOL))
    assert np.all(result <= (y + TOL))


def test_Div_a_is_op():
    op = Div(Input((-1, 5), np.dtype(np.float32)), 2.0)
    tf_op = TensorflowConverter().visit(op)
    x = np.array([[1.0, 2.0, 3.0, 4.0, 5.0]], dtype=np.float32)
    result = tf_op(x).numpy()
    y = np.array([[0.5, 1.0, 1.5, 2.0, 2.5]], dtype=np.float32)
    assert np.all(result >= (y - TOL))
    assert np.all(result <= (y + TOL))


def test_Div_b_is_op():
    op = Div(10.0, Input((-1, 5), np.dtype(np.float32)))
    tf_op = TensorflowConverter().visit(op)
    x = np.array([[-1.0, -2.0, -3.0, -4.0, -5.0]], dtype=np.float32)
    result = tf_op(x).numpy()
    y = np.array([[-10.0, -5.0, -3.3333333333, -2.5, -2]], dtype=np.float32)
    assert np.all(result >= (y - TOL))
    assert np.all(result <= (y + TOL))


def test_Div_a_b_are_ops():
    op = Div(Input((-1, 5), np.dtype(np.float32)), Input((-1, 5), np.dtype(np.float32)))
    tf_op = TensorflowConverter().visit(op)
    a = np.array([[-1.0, -2.0, -3.0, -4.0, -5.0]], dtype=np.float32)
    b = np.array([[1.0, 2.0, 3.0, 4.0, 5.0]], dtype=np.float32)
    result = tf_op(a, b).numpy()
    y = -np.ones((1, 5), dtype=np.float32)
    assert np.all(result >= (y - TOL))
    assert np.all(result <= (y + TOL))
