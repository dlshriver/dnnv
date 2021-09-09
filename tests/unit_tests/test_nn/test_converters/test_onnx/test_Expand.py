import numpy as np
import onnxruntime

from dnnv.nn.converters.onnx import *
from dnnv.nn.operations import *


def test_Expand_dim_changed():
    shape = np.array([3, 1])
    data = np.reshape(np.arange(1, np.prod(shape) + 1, dtype=np.float32), shape)
    new_shape = np.array([2, 1, 6])

    op = Expand(data, new_shape)
    onnx_model = convert(OperationGraph([op]))

    results = onnxruntime.backend.run(onnx_model, [])
    assert len(results) == 1
    result = results[0]

    y = data * np.ones(new_shape, dtype=np.float32)

    assert result.shape == y.shape
    assert np.allclose(result, y)


def test_Expand_dim_unchanged():
    shape = np.array([3, 1])
    new_shape = np.array([3, 4])
    data = np.reshape(np.arange(1, np.prod(shape) + 1, dtype=np.float32), shape)

    input_op = Input(shape, np.dtype(np.float32))
    op = Expand(input_op, new_shape)
    onnx_model = convert(OperationGraph([op]))

    results = onnxruntime.backend.run(onnx_model, [data])
    assert len(results) == 1
    result = results[0]

    y = np.tile(data, 4)

    assert result.shape == y.shape
    assert np.allclose(result, y)
