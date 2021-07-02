import numpy as np
import pytest

from dnnv.nn.converters.tensorflow import *
from dnnv.nn.operations import *


def test_Tile():
    x = np.random.rand(2, 3, 4, 5).astype(np.float32)
    repeats = np.random.randint(low=1, high=10, size=(np.ndim(x),)).astype(np.int64)
    z = np.tile(x, repeats)

    op = Tile(x, repeats)
    tf_op = TensorflowConverter().visit(op)
    result = tf_op().numpy()
    assert np.allclose(result, z)

    op = Tile(
        Input((2, 3, 4, 5), np.dtype(np.float32)),
        Input((4,), np.dtype(np.int64)),
    )
    tf_op = TensorflowConverter().visit(op)
    result = tf_op(x, repeats).numpy()
    assert np.allclose(result, z)


def test_Tile_precomputed():
    x = np.array([[0, 1], [2, 3]], dtype=np.float32)
    repeats = np.array([2, 2], dtype=np.int64)
    z = np.array(
        [[0, 1, 0, 1], [2, 3, 2, 3], [0, 1, 0, 1], [2, 3, 2, 3]], dtype=np.float32
    )

    op = Tile(x, repeats)
    tf_op = TensorflowConverter().visit(op)
    result = tf_op().numpy()
    assert np.allclose(result, z)
