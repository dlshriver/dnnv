import numpy as np
import onnx

from dnnv.nn.converters.tensorflow import *
from dnnv.nn.operations import *


def test_Cast_consts():
    x = np.arange(12).reshape((1, 3, 2, 2))

    op = Cast(x, onnx.TensorProto.FLOAT)
    tf_op = TensorflowConverter().visit(op)
    result = tf_op().numpy()
    assert result.dtype == np.float32


def test_Cast_x_is_op():
    x = np.arange(12).reshape((1, 3, 2, 2))
    scale = np.full(3, 2.0, dtype=np.float32)
    bias = np.full(3, 0.0, dtype=np.float32)
    mean = np.full(3, 5.5, dtype=np.float32)
    var = np.full(3, 11.9, dtype=np.float32)

    input_op = Input((1, 3, 2, 2), np.dtype(np.int64))
    op = Cast(input_op, onnx.TensorProto.FLOAT)
    tf_op = TensorflowConverter().visit(op)
    result = tf_op(x).numpy()
    assert result.dtype == np.float32
