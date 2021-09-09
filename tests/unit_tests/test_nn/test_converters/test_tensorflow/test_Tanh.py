import numpy as np
import pytest

from dnnv.nn.converters.tensorflow import *
from dnnv.nn.operations import *


def test_Tanh():
    x = np.array([-1, 0, 1]).astype(np.float32)
    y = np.tanh(x)

    op = Tanh(x)
    tf_op = TensorflowConverter().visit(op)
    result = tf_op().numpy()
    assert np.allclose(result, y)

    op = Tanh(Input((3,), np.dtype(np.float32)))
    tf_op = TensorflowConverter().visit(op)
    result = tf_op(x).numpy()
    assert np.allclose(result, y)


def test_Tanh_rand():
    x = np.random.randn(3, 4, 5).astype(np.float32)
    y = np.tanh(x)

    op = Tanh(x)
    tf_op = TensorflowConverter().visit(op)
    result = tf_op().numpy()
    assert np.allclose(result, y)
