import numpy as np
import pytest

from dnnv.nn.converters.tensorflow import *
from dnnv.nn.operations import *


def test_Identity():
    data = np.array(
        [
            [
                [
                    [1, 2],
                    [3, 4],
                ]
            ]
        ],
        dtype=np.float32,
    )
    op = Identity(data)
    tf_op = TensorflowConverter().visit(op)
    result = tf_op().numpy()
    y = data
    assert result.shape == y.shape
    assert np.allclose(result, y)

    input_op = Input((1, 1, 2, 2), np.dtype(np.float32))
    op = Identity(input_op)
    tf_op = TensorflowConverter().visit(op)
    result = tf_op(data).numpy()
    y = data
    assert result.shape == y.shape
    assert np.allclose(result, y)
