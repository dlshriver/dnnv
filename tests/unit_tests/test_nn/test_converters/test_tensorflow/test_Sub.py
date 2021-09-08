import numpy as np
import pytest

from dnnv.nn.converters.tensorflow import *
from dnnv.nn.operations import *


def test_Sub():
    x = np.array([1, 2, 3]).astype(np.float32)
    y = np.array([3, 2, 1]).astype(np.float32)
    z = x - y

    op = Sub(x, y)
    tf_op = TensorflowConverter().visit(op)
    result = tf_op().numpy()
    assert np.allclose(result, z)

    op = Sub(Input((3,), np.dtype(np.float32)), Input((3,), np.dtype(np.float32)))
    tf_op = TensorflowConverter().visit(op)
    result = tf_op(x, y).numpy()
    assert np.allclose(result, z)


def test_Sub_rand():
    x = np.random.randn(3, 4, 5).astype(np.float32)
    y = np.random.randn(3, 4, 5).astype(np.float32)
    z = x - y

    op = Sub(x, y)
    tf_op = TensorflowConverter().visit(op)
    result = tf_op().numpy()
    assert np.allclose(result, z)

    x = np.random.randint(12, 24, size=(3, 4, 5), dtype=np.uint8)
    y = np.random.randint(12, size=(3, 4, 5), dtype=np.uint8)
    z = x - y
    op = Sub(x, y)
    tf_op = TensorflowConverter().visit(op)
    result = tf_op().numpy()
    assert np.allclose(result, z)


def test_Sub_broadcast():
    x = np.random.randn(3, 4, 5).astype(np.float32)
    y = np.random.randn(5).astype(np.float32)
    z = x - y

    op = Sub(x, y)
    tf_op = TensorflowConverter().visit(op)
    result = tf_op().numpy()
    assert np.allclose(result, z)
