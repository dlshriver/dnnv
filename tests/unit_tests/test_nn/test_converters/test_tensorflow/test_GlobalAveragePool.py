import numpy as np
import pytest

from dnnv.nn.converters.tensorflow import *
from dnnv.nn.operations import *


def test_GlobalAveragePool():
    x = np.random.randn(1, 3, 5, 5).astype(np.float32)
    op = GlobalAveragePool(x)
    tf_op = TensorflowConverter().visit(op)
    result = tf_op().numpy()
    y = np.mean(x, axis=tuple(range(2, np.ndim(x))), keepdims=True)
    assert result.shape == y.shape
    assert np.allclose(result, y)

    x = np.random.randn(1, 3, 5, 5).astype(np.float32)
    input_op = Input((1, 3, 5, 5), np.dtype(np.float32))
    op = GlobalAveragePool(input_op)
    tf_op = TensorflowConverter().visit(op)
    result = tf_op(x).numpy()
    y = np.mean(x, axis=tuple(range(2, np.ndim(x))), keepdims=True)
    assert result.shape == y.shape
    assert np.allclose(result, y)


def test_GlobalAveragePool_precomputed():
    x = np.array(
        [
            [
                [
                    [1, 2, 3],
                    [4, 5, 6],
                    [7, 8, 9],
                ]
            ]
        ]
    ).astype(np.float32)
    op = GlobalAveragePool(x)
    tf_op = TensorflowConverter().visit(op)
    result = tf_op().numpy()
    y = np.array([[[[5]]]]).astype(np.float32)
    assert result.shape == y.shape
    assert np.allclose(result, y)
