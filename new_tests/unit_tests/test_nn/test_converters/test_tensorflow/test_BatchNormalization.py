import numpy as np

from dnnv.nn.converters.tensorflow import *
from dnnv.nn.operations import *

TOL = 1e-6


def test_BatchNormalization_consts():
    x = np.arange(12).astype(np.float32).reshape((1, 3, 2, 2))
    scale = np.full(3, 2.0, dtype=np.float32)
    bias = np.full(3, 0.0, dtype=np.float32)
    mean = np.full(3, 5.5, dtype=np.float32)
    var = np.full(3, 11.9, dtype=np.float32)

    op = BatchNormalization(x, scale, bias, mean, var)
    tf_op = TensorflowConverter().visit(op)
    result = tf_op().numpy()
    y = np.array(
        [
            [
                [[-3.1887393, -2.6089685], [-2.0291977, -1.4494269]],
                [[-0.8696561, -0.28988528], [0.28988552, 0.8696561]],
                [[1.4494271, 2.0291982], [2.6089687, 3.1887393]],
            ]
        ],
        dtype=np.float32,
    )
    assert np.all(result >= (y - TOL))
    assert np.all(result <= (y + TOL))


def test_BatchNormalization_x_is_op():
    x = np.arange(12).astype(np.float32).reshape((1, 3, 2, 2))
    scale = np.full(3, 2.0, dtype=np.float32)
    bias = np.full(3, 0.0, dtype=np.float32)
    mean = np.full(3, 5.5, dtype=np.float32)
    var = np.full(3, 11.9, dtype=np.float32)

    input_op = Input((1, 3, 2, 2), np.dtype(np.float32))
    op = BatchNormalization(input_op, scale, bias, mean, var)
    tf_op = TensorflowConverter().visit(op)
    result = tf_op(x).numpy()
    y = np.array(
        [
            [
                [[-3.1887393, -2.6089685], [-2.0291977, -1.4494269]],
                [[-0.8696561, -0.28988528], [0.28988552, 0.8696561]],
                [[1.4494271, 2.0291982], [2.6089687, 3.1887393]],
            ]
        ],
        dtype=np.float32,
    )
    assert np.all(result >= (y - TOL))
    assert np.all(result <= (y + TOL))
