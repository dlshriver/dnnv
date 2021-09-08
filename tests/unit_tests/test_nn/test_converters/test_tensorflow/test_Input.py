import numpy as np
import pytest

from dnnv.nn.converters.tensorflow import *
from dnnv.nn.operations import *


def test_Input():
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
    op = Input((1, 1, 2, 2), np.dtype(np.float32))
    tf_op = TensorflowConverter().visit(op)
    result = tf_op(data).numpy()
    y = data
    assert result.shape == y.shape
    assert np.allclose(result, y)


def test_Input_incorrect_shape():
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
    op = Input((1, 4), np.dtype(np.float32))
    tf_op = TensorflowConverter().visit(op)
    with pytest.raises(TensorflowConverterError, match="Incorrect input shape: .*"):
        _ = tf_op(data).numpy()


def test_Input_incorrect_dtype():
    data = np.array(
        [
            [
                [
                    [1, 2],
                    [3, 4],
                ]
            ]
        ],
        dtype=np.float64,
    )
    op = Input((1, 1, 2, 2), np.dtype(np.float32))
    tf_op = TensorflowConverter().visit(op)
    with pytest.raises(TensorflowConverterError, match="Incorrect type, .*"):
        _ = tf_op(data).numpy()
