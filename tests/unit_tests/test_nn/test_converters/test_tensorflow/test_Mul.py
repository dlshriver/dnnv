import numpy as np
import pytest

from dnnv.nn.converters.tensorflow import *
from dnnv.nn.operations import *


def test_Mul():
    x = np.array([1, 2, 3]).astype(np.float32)
    y = np.array([4, 5, 6]).astype(np.float32)
    z = x * y  # expected output [4., 10., 18.]

    op = Mul(x, y)
    tf_op = TensorflowConverter().visit(op)
    result = tf_op().numpy()
    assert np.allclose(result, z)

    op = Mul(Input((3,), np.dtype(np.float32)), Input((3,), np.dtype(np.float32)))
    tf_op = TensorflowConverter().visit(op)
    result = tf_op(x, y).numpy()
    assert np.allclose(result, z)


def test_Mul_rand():
    x = np.random.randn(3, 4, 5).astype(np.float32)
    y = np.random.randn(3, 4, 5).astype(np.float32)
    z = x * y

    op = Mul(x, y)
    tf_op = TensorflowConverter().visit(op)
    result = tf_op().numpy()
    assert np.allclose(result, z)

    x = np.random.randint(4, size=(3, 4, 5), dtype=np.uint8)
    y = np.random.randint(24, size=(3, 4, 5), dtype=np.uint8)
    z = x * y
    op = Mul(x, y)
    tf_op = TensorflowConverter().visit(op)
    result = tf_op().numpy()
    assert np.allclose(result, z)


def test_Mul_broadcast():
    x = np.random.randn(3, 4, 5).astype(np.float32)
    y = np.random.randn(5).astype(np.float32)
    z = x * y

    op = Mul(x, y)
    tf_op = TensorflowConverter().visit(op)
    result = tf_op().numpy()
    assert np.allclose(result, z)
