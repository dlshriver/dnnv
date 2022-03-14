import numpy as np
import onnxruntime.backend

from dnnv.nn.converters.onnx import OnnxConverter
from dnnv.nn.graph import OperationGraph
from dnnv.nn.operations import *


def test_Sign():
    x = np.array(range(-5, 6)).astype(np.float32)
    y = np.sign(x)

    op = Sign(x)
    onnx_model = OnnxConverter.convert(OperationGraph([op]))
    result = onnxruntime.backend.run(onnx_model, [])[0]
    assert np.allclose(result, y)

    op = Sign(Input((11,), np.dtype(np.float32)))
    onnx_model = OnnxConverter.convert(OperationGraph([op]))
    result = onnxruntime.backend.run(onnx_model, [x])[0]
    assert np.allclose(result, y)
