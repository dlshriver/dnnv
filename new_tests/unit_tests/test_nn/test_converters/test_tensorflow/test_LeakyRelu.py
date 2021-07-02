import numpy as np
import pytest

from dnnv.nn.converters.tensorflow import *
from dnnv.nn.operations import *


def test_LeakyRelu():
    x = np.array([-1, 0, 1]).astype(np.float32)
    op = LeakyRelu(x, alpha=0.1)
    tf_op = TensorflowConverter().visit(op)
    result = tf_op().numpy()
    y = np.clip(x, 0, np.inf) + np.clip(x, -np.inf, 0) * 0.1
    assert np.allclose(result, y)


def test_LeakyRelu_random():
    x = np.random.randn(3, 4, 5).astype(np.float32)
    input_op = Input((3, 4, 5), np.dtype(np.float32))
    op = LeakyRelu(input_op, alpha=0.1)
    tf_op = TensorflowConverter().visit(op)
    result = tf_op(x).numpy()
    y = np.clip(x, 0, np.inf) + np.clip(x, -np.inf, 0) * 0.1
    assert np.allclose(result, y)


def test_default():
    x = np.random.randn(3, 4, 5).astype(np.float32)
    op = LeakyRelu(x)
    tf_op = TensorflowConverter().visit(op)
    result = tf_op().numpy()
    y = np.clip(x, 0, np.inf) + np.clip(x, -np.inf, 0) * 0.01
    assert np.allclose(result, y)
