import numpy as np
import onnxruntime.backend

from dnnv.nn.converters.onnx import OnnxConverter
from dnnv.nn.graph import OperationGraph
from dnnv.nn.operations import *


def test_Shape():
    x = np.array(
        [
            [1, 2, 3],
            [4, 5, 6],
        ]
    ).astype(np.float32)

    op = Shape(x)
    onnx_model = OnnxConverter.convert(OperationGraph([op]))

    results = onnxruntime.backend.run(onnx_model, [])
    assert len(results) == 1
    result = results[0]
    assert np.allclose(result, np.array([2, 3]))

    op = Shape(Input((1, 17), np.float64))
    onnx_model = OnnxConverter.convert(OperationGraph([op]))

    results = onnxruntime.backend.run(onnx_model, [np.random.rand(1, 17)])
    assert len(results) == 1
    result = results[0]
    assert np.allclose(result, np.array([1, 17]))
