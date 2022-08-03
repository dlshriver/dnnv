import numpy as np

from dnnv.nn.converters.tensorflow import *
from dnnv.nn.operations import *


def test_Squeeze():
    x = np.random.randn(1, 3, 4, 5).astype(np.float32)
    axes = 0
    y = np.squeeze(x, axis=axes)

    op = Squeeze(x, axes=axes)
    tf_op = TensorflowConverter().visit(op)
    result = tf_op().numpy()
    assert np.all(result == y)


def test_Squeeze_negative_axes():
    x = np.random.randn(1, 3, 1, 5).astype(np.float32)
    axes = -2
    y = np.squeeze(x, axis=axes)

    op = Squeeze(x, axes=axes)
    tf_op = TensorflowConverter().visit(op)
    result = tf_op().numpy()
    assert np.all(result == y)
