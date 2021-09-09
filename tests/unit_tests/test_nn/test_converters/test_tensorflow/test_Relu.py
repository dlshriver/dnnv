import numpy as np
import pytest

from dnnv.nn.converters.tensorflow import *
from dnnv.nn.operations import *


def test_Relu():
    x = np.random.randn(3, 4, 5).astype(np.float32)
    y = np.clip(x, 0, np.inf)

    op = Relu(x)
    tf_op = TensorflowConverter().visit(op)
    result = tf_op().numpy()
    assert np.allclose(result, y)

    op = Relu(Input((3, 4, 5), np.dtype(np.float32)))
    tf_op = TensorflowConverter().visit(op)
    result = tf_op(x).numpy()
    assert np.allclose(result, y)
