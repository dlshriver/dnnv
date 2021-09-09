import numpy as np
import onnxruntime

from dnnv.nn.converters.onnx import *
from dnnv.nn.operations import *


def test_Sigmoid():
    x = np.array([-1, 0, 1]).astype(np.float32)
    y = 1.0 / (1.0 + np.exp(np.negative(x)))

    op = Sigmoid(x)
    onnx_model = convert(OperationGraph([op]))

    results = onnxruntime.backend.run(onnx_model, [])
    assert len(results) == 1
    result = results[0]
    assert np.allclose(result, y)

    op = Sigmoid(Input((3,), np.dtype(np.float32)))
    onnx_model = convert(OperationGraph([op]))

    results = onnxruntime.backend.run(onnx_model, [x])
    assert len(results) == 1
    result = results[0]
    assert np.allclose(result, y)


def test_Sigmoid_rand():
    x = np.random.randn(3, 4, 5).astype(np.float32)
    y = 1.0 / (1.0 + np.exp(np.negative(x)))

    op = Sigmoid(x)
    onnx_model = convert(OperationGraph([op]))

    results = onnxruntime.backend.run(onnx_model, [])
    assert len(results) == 1
    result = results[0]
    assert np.allclose(result, y)
