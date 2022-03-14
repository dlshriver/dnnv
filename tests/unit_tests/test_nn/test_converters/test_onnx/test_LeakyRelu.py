import numpy as np
import onnxruntime.backend

from dnnv.nn.converters.onnx import OnnxConverter
from dnnv.nn.graph import OperationGraph
from dnnv.nn.operations import *


def test_LeakyRelu():
    x = np.array([-1, 0, 1]).astype(np.float32)
    op = LeakyRelu(x, alpha=0.1)
    onnx_model = OnnxConverter.convert(OperationGraph([op]))
    result = onnxruntime.backend.run(onnx_model, [])[0]
    y = np.clip(x, 0, np.inf) + np.clip(x, -np.inf, 0) * 0.1
    assert np.allclose(result, y)


def test_LeakyRelu_random():
    x = np.random.randn(3, 4, 5).astype(np.float32)
    input_op = Input((3, 4, 5), np.dtype(np.float32))
    op = LeakyRelu(input_op, alpha=0.1)
    onnx_model = OnnxConverter.convert(OperationGraph([op]))
    result = onnxruntime.backend.run(onnx_model, [x])[0]
    y = np.clip(x, 0, np.inf) + np.clip(x, -np.inf, 0) * 0.1
    assert np.allclose(result, y)


def test_default():
    x = np.random.randn(3, 4, 5).astype(np.float32)
    op = LeakyRelu(x)
    onnx_model = OnnxConverter.convert(OperationGraph([op]))
    result = onnxruntime.backend.run(onnx_model, [])[0]
    y = np.clip(x, 0, np.inf) + np.clip(x, -np.inf, 0) * 0.01
    assert np.allclose(result, y)
