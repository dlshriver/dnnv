import numpy as np
import pytest

from dnnv.nn.converters.tensorflow import *
from dnnv.nn.operations import *

TOL = 1e-6


def test_Elu_consts():
    x = np.array([-1, 0, 1]).astype(np.float32)
    op = Elu(x, alpha=2.0)
    tf_op = TensorflowConverter().visit(op)
    with pytest.raises(TensorflowConverterError):
        result = tf_op().numpy()
    # y = np.clip(x, 0, np.inf) + (np.exp(np.clip(x, -np.inf, 0)) - 1) * 2.0
    # assert np.all(result >= (y - TOL))
    # assert np.all(result <= (y + TOL))


def test_Elu_default_consts():
    x = np.random.randn(3, 4, 5).astype(np.float32)
    op = Elu(x)
    tf_op = TensorflowConverter().visit(op)
    result = tf_op().numpy()
    y = np.clip(x, 0, np.inf) + (np.exp(np.clip(x, -np.inf, 0)) - 1)
    assert np.all(result >= (y - TOL))
    assert np.all(result <= (y + TOL))


def test_Elu_default_x_is_op():
    x = np.random.randn(3, 4, 5).astype(np.float32)
    input_op = Input((3, 4, 5), np.dtype(np.float32))
    op = Elu(input_op)
    tf_op = TensorflowConverter().visit(op)
    result = tf_op(x).numpy()
    y = np.clip(x, 0, np.inf) + (np.exp(np.clip(x, -np.inf, 0)) - 1)
    assert np.all(result >= (y - TOL))
    assert np.all(result <= (y + TOL))
