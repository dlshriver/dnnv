import numpy as np
import onnxruntime.backend

from dnnv.nn.converters.onnx import OnnxConverter
from dnnv.nn.graph import OperationGraph
from dnnv.nn.operations import *


def test_RandomUniform():
    op = RandomUniform((1, 5))
    onnx_model = OnnxConverter().convert(OperationGraph([op]))

    results = onnxruntime.backend.run(onnx_model, [])
    assert len(results) == 1
    result = results[0]
    assert result.shape == (1, 5)
    assert result.dtype == np.float32
    assert np.all(result >= 0)
    assert np.all(result <= 1)

    op = RandomUniform((1, 3, 2, 2), dtype=np.float64, low=101.1, high=314.0, seed=7.0)
    onnx_model = OnnxConverter().convert(OperationGraph([op]))

    results = onnxruntime.backend.run(onnx_model, [])
    assert len(results) == 1
    result = results[0]
    assert result.shape == (1, 3, 2, 2)
    assert result.dtype == np.float64
    assert np.all(result >= 101.1)
    assert np.all(result <= 314.0)
