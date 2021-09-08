import numpy as np
import onnxruntime

from dnnv.nn.converters.onnx import *
from dnnv.nn.operations import *


def test_GlobalAveragePool():
    x = np.random.randn(1, 3, 5, 5).astype(np.float32)
    op = GlobalAveragePool(x)
    onnx_model = convert(OperationGraph([op]))

    results = onnxruntime.backend.run(onnx_model, [])
    assert len(results) == 1
    result = results[0]

    y = np.mean(x, axis=tuple(range(2, np.ndim(x))), keepdims=True)
    assert result.shape == y.shape
    assert np.allclose(result, y)

    x = np.random.randn(1, 3, 5, 5).astype(np.float32)
    input_op = Input((1, 3, 5, 5), np.dtype(np.float32))
    op = GlobalAveragePool(input_op)
    onnx_model = convert(OperationGraph([op]))

    results = onnxruntime.backend.run(onnx_model, [x])
    assert len(results) == 1
    result = results[0]

    y = np.mean(x, axis=tuple(range(2, np.ndim(x))), keepdims=True)
    assert result.shape == y.shape
    assert np.allclose(result, y)


def test_GlobalAveragePool_precomputed():
    x = np.array(
        [
            [
                [
                    [1, 2, 3],
                    [4, 5, 6],
                    [7, 8, 9],
                ]
            ]
        ]
    ).astype(np.float32)
    op = GlobalAveragePool(x)
    onnx_model = convert(OperationGraph([op]))

    results = onnxruntime.backend.run(onnx_model, [])
    assert len(results) == 1
    result = results[0]

    y = np.array([[[[5]]]]).astype(np.float32)
    assert result.shape == y.shape
    assert np.allclose(result, y)
