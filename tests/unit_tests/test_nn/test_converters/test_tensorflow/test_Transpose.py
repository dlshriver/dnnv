import itertools
import numpy as np
import pytest

from dnnv.nn.converters.tensorflow import *
from dnnv.nn.operations import *


def test_Transpose():
    shape = (2, 3, 4)
    data = np.random.random_sample(shape).astype(np.float32)
    z = np.transpose(data)

    op = Transpose(data)
    tf_op = TensorflowConverter().visit(op)
    result = tf_op().numpy()
    assert np.allclose(result, z)

    op = Transpose(
        Input(shape, np.dtype(np.float32)),
    )
    tf_op = TensorflowConverter().visit(op)
    result = tf_op(data).numpy()
    assert np.allclose(result, z)


def test_Transpose_all_permutations():
    shape = (2, 3, 4)
    data = np.random.random_sample(shape).astype(np.float32)
    permutations = list(itertools.permutations(np.arange(len(shape))))

    for permutation in permutations:
        z = np.transpose(data, permutation)
        op = Transpose(data, permutation=np.asarray(permutation))
        tf_op = TensorflowConverter().visit(op)
        result = tf_op().numpy()
        assert np.allclose(result, z)
