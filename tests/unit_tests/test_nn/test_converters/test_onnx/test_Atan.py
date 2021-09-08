import numpy as np
import onnxruntime

from dnnv.nn.converters.onnx import *
from dnnv.nn.operations import *


def test_Atan_const():
    op = Atan(3.0)
    onnx_model = convert(OperationGraph([op]))

    results = onnxruntime.backend.run(onnx_model, [])
    assert len(results) == 1
    result = results[0]
    assert np.allclose(result, 1.2490457)


def test_Atan_x_is_op():
    op = Atan(Input((1, 5), np.dtype(np.float32)))
    onnx_model = convert(OperationGraph([op]))

    x = np.array([[-0.2, -0.1, 0.0, 0.1, 0.2]], dtype=np.float32)
    results = onnxruntime.backend.run(onnx_model, [x])
    assert len(results) == 1
    result = results[0]

    y = np.array(
        [[-0.19739556, -0.09966865, 0.0, 0.09966865, 0.19739556]], dtype=np.float32
    )
    assert np.allclose(result, y)
