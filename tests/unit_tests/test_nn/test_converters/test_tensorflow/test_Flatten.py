import numpy as np

from dnnv.nn.converters.tensorflow import *
from dnnv.nn.operations import *

TOL = 1e-6


def test_Flatten():
    shape = (2, 3, 4, 5)
    a = np.random.random_sample(shape).astype(np.float32)

    for i in range(a.ndim):
        axis = i
        op = Flatten(a, axis=axis)
        tf_op = TensorflowConverter().visit(op)
        result = tf_op().numpy()

        new_shape = (1, -1) if axis == 0 else (np.prod(shape[0:axis]).astype(int), -1)
        y = np.reshape(a, new_shape)

        assert result.shape == y.shape
        assert np.all(result >= (y - TOL))
        assert np.all(result <= (y + TOL))


def test_Flatten_negative_axis():
    shape = (2, 3, 4, 5)
    a = np.random.random_sample(shape).astype(np.float32)

    for i in range(a.ndim):
        axis = i - 1
        op = Flatten(a, axis=axis)
        tf_op = TensorflowConverter().visit(op)
        result = tf_op().numpy()

        new_shape = (np.prod(shape[0:axis]).astype(int), -1)
        y = np.reshape(a, new_shape)

        assert result.shape == y.shape
        assert np.all(result >= (y - TOL))
        assert np.all(result <= (y + TOL))


def test_Flatten_with_default_axis():
    shape = (5, 4, 3, 2)
    a = np.random.random_sample(shape).astype(np.float32)

    input_op = Input(shape, np.dtype(np.float32))
    op = Flatten(input_op)
    tf_op = TensorflowConverter().visit(op)
    result = tf_op(a).numpy()

    new_shape = (5, 24)
    y = np.reshape(a, new_shape)

    assert result.shape == y.shape
    assert np.all(result >= (y - TOL))
    assert np.all(result <= (y + TOL))
