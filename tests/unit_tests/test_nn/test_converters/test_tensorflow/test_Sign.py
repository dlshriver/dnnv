import numpy as np
import pytest

from dnnv.nn.converters.tensorflow import *
from dnnv.nn.operations import *


def test_Sign():
    x = np.array(range(-5, 6)).astype(np.float32)
    y = np.sign(x)

    op = Sign(x)
    tf_op = TensorflowConverter().visit(op)
    result = tf_op().numpy()
    assert np.allclose(result, y)

    op = Sign(Input((12,), np.dtype(np.float32)))
    tf_op = TensorflowConverter().visit(op)
    result = tf_op(x).numpy()
    assert np.allclose(result, y)
