import numpy as np
import onnxruntime.backend

from dnnv.nn.converters.onnx import *
from dnnv.nn.operations import *


def test_Squeeze():
    x = np.random.randn(1, 3, 4, 5).astype(np.float32)
    axes = np.array([0], dtype=np.int64)
    y = np.squeeze(x, axis=0)

    op = Squeeze(x, axes=axes)
    onnx_model = convert(OperationGraph([op]))
    output = onnxruntime.backend.run(onnx_model, [])
    assert np.all(output == y)


def test_Squeeze_negative_axes():
    x = np.random.randn(1, 3, 1, 5).astype(np.float32)
    axes = np.array([-2], dtype=np.int64)
    y = np.squeeze(x, axis=-2)

    op = Squeeze(x, axes=axes)
    onnx_model = convert(OperationGraph([op]))
    output = onnxruntime.backend.run(onnx_model, [x, axes])
    assert np.all(output == y)
