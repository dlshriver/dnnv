import numpy as np

from dnnv.nn.converters.tensorflow import *
from dnnv.nn.operations import *

TOL = 1e-6


def test_Atan_const():
    op = Atan(3.0)
    tf_op = TensorflowConverter().visit(op)
    result = tf_op().numpy()
    assert result > (1.2490457 - TOL)
    assert result < (1.2490457 + TOL)


def test_Atan_x_is_op():
    op = Atan(Input((1, 5), np.dtype(np.float32)))
    tf_op = TensorflowConverter().visit(op)
    x = np.array([[-0.2, -0.1, 0.0, 0.1, 0.2]], dtype=np.float32)
    result = tf_op(x).numpy()
    y = np.array(
        [[-0.19739556, -0.09966865, 0.0, 0.09966865, 0.19739556]], dtype=np.float32
    )
    assert np.all(result >= (y - TOL))
    assert np.all(result <= (y + TOL))
