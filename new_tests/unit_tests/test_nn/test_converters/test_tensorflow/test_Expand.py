import numpy as np

from dnnv.nn.converters.tensorflow import *
from dnnv.nn.operations import *

TOL = 1e-6


def test_Expand_dim_changed():
    shape = [3, 1]
    data = np.reshape(np.arange(1, np.prod(shape) + 1, dtype=np.float32), shape)
    new_shape = [2, 1, 6]

    op = Expand(data, new_shape)
    tf_op = TensorflowConverter().visit(op)
    result = tf_op().numpy()

    y = data * np.ones(new_shape, dtype=np.float32)

    assert result.shape == y.shape
    assert np.all(result >= (y - TOL))
    assert np.all(result <= (y + TOL))


def test_Expand_dim_unchanged():
    shape = [3, 1]
    new_shape = [3, 4]
    data = np.reshape(np.arange(1, np.prod(shape) + 1, dtype=np.float32), shape)

    input_op = Input(shape, np.dtype(np.float32))
    op = Expand(input_op, new_shape)
    tf_op = TensorflowConverter().visit(op)
    result = tf_op(data).numpy()

    y = np.tile(data, 4)

    assert result.shape == y.shape
    assert np.all(result >= (y - TOL))
    assert np.all(result <= (y + TOL))
