import numpy as np
import onnxruntime
import pytest

from dnnv.nn.converters.onnx import *
from dnnv.nn.operations import *


@pytest.mark.xfail
def test_Elu_consts():
    x = np.array([-1, 0, 1]).astype(np.float32)
    op = Elu(x, alpha=2.0)
    onnx_model = convert(OperationGraph([op]))

    results = onnxruntime.backend.run(onnx_model, [])
    assert len(results) == 1
    result = results[0]

    y = np.clip(x, 0, np.inf) + (np.exp(np.clip(x, -np.inf, 0)) - 1) * 2.0
    assert np.allclose(result, y)


def test_Elu_default_consts():
    x = np.random.randn(3, 4, 5).astype(np.float32)
    op = Elu(x)
    onnx_model = convert(OperationGraph([op]))

    results = onnxruntime.backend.run(onnx_model, [])
    assert len(results) == 1
    result = results[0]

    y = np.clip(x, 0, np.inf) + (np.exp(np.clip(x, -np.inf, 0)) - 1)
    assert np.allclose(result, y)


def test_Elu_default_x_is_op():
    x = np.random.randn(3, 4, 5).astype(np.float32)
    input_op = Input((3, 4, 5), np.dtype(np.float32))
    op = Elu(input_op)
    onnx_model = convert(OperationGraph([op]))

    results = onnxruntime.backend.run(onnx_model, [x])
    assert len(results) == 1
    result = results[0]

    y = np.clip(x, 0, np.inf) + (np.exp(np.clip(x, -np.inf, 0)) - 1)
    assert np.allclose(result, y)
