import numpy as np

from dnnv.nn.converters.tensorflow import *
from dnnv.nn.operations import *

TOL = 1e-6


def test_Dropout_consts():
    x = np.array([3, 4]).astype(np.float32)
    op = Dropout(x)
    tf_op = TensorflowConverter().visit(op)
    result = tf_op()
    y = np.array([3, 4]).astype(np.float32)
    assert np.all(result >= (y - TOL))
    assert np.all(result <= (y + TOL))


def test_Dropout_x_is_op():
    x = np.array([3, 4]).astype(np.float32)
    input_op = Input((2,), np.dtype(np.float32))
    op = Dropout(input_op)
    tf_op = TensorflowConverter().visit(op)
    result = tf_op(x)
    y = np.array([3, 4]).astype(np.float32)
    assert np.all(result >= (y - TOL))
    assert np.all(result <= (y + TOL))
