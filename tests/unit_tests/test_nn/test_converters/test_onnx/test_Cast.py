import numpy as np
import onnx
import onnxruntime

from dnnv.nn.converters.onnx import *
from dnnv.nn.operations import *


def test_Cast_consts():
    x = np.arange(12).reshape((1, 3, 2, 2))

    op = Cast(x, onnx.TensorProto.FLOAT)
    onnx_model = convert(OperationGraph([op]))

    results = onnxruntime.backend.run(onnx_model, [])
    assert len(results) == 1
    result = results[0]
    assert result.dtype == np.float32


def test_Cast_x_is_op():
    x = np.arange(12).reshape((1, 3, 2, 2))

    input_op = Input((1, 3, 2, 2), np.dtype(np.int64))
    op = Cast(input_op, onnx.TensorProto.FLOAT)
    onnx_model = convert(OperationGraph([op]))

    results = onnxruntime.backend.run(onnx_model, [x])
    assert len(results) == 1
    result = results[0]
    assert result.dtype == np.float32
