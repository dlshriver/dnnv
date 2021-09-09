import numpy as np
import pytest

from dnnv.nn.converters.tensorflow import *
from dnnv.nn.operations import *


def test_Shape():
    x = np.array(
        [
            [1, 2, 3],
            [4, 5, 6],
        ]
    ).astype(np.float32)
    y = np.array(
        [
            2,
            3,
        ]
    ).astype(np.int64)

    op = Shape(x)
    tf_op = TensorflowConverter().visit(op)
    result = tf_op().numpy()
    assert np.allclose(result, y)

    op = Shape(
        Input((2, 3), np.dtype(np.float32)),
    )
    tf_op = TensorflowConverter().visit(op)
    result = tf_op(x).numpy()
    assert np.allclose(result, y)


def test_Shape_rand():
    x = np.random.randn(3, 4, 5).astype(np.float32)
    y = np.array(x.shape).astype(np.int64)

    op = Shape(x)
    tf_op = TensorflowConverter().visit(op)
    result = tf_op().numpy()
    assert np.allclose(result, y)
