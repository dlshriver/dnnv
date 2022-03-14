import numpy as np

from dnnv.nn.converters.tensorflow import TensorflowConverter
from dnnv.nn.graph import OperationGraph
from dnnv.nn.operations import *


def test_RandomUniform():
    op = RandomUniform((1, 5))
    tf_op = TensorflowConverter().convert(OperationGraph([op]))

    result = tf_op()
    assert result.shape == (1, 5)
    assert result.dtype == np.float32
    assert np.all(result >= 0)
    assert np.all(result <= 1)

    op = RandomUniform((1, 3, 2, 2), dtype=np.float64, low=101.1, high=314.0, seed=7.0)
    tf_op = TensorflowConverter().convert(OperationGraph([op]))

    result = tf_op()
    assert result.shape == (1, 3, 2, 2)
    assert result.dtype == np.float64
    assert np.all(result >= 101.1)
    assert np.all(result <= 314.0)
