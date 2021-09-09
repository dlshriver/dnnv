import numpy as np
import onnxruntime
import pytest

from dnnv.nn.converters.onnx import *
from dnnv.nn.operations import *


def test_Sub():
    x = np.array([1, 2, 3]).astype(np.float32)
    y = np.array([3, 2, 1]).astype(np.float32)
    z = x - y

    op = Sub(x, y)
    onnx_model = convert(OperationGraph([op]))

    results = onnxruntime.backend.run(onnx_model, [])
    assert len(results) == 1
    result = results[0]
    assert np.allclose(result, z)

    op = Sub(Input((3,), np.dtype(np.float32)), Input((3,), np.dtype(np.float32)))
    onnx_model = convert(OperationGraph([op]))

    results = onnxruntime.backend.run(onnx_model, [x, y])
    assert len(results) == 1
    result = results[0]
    assert np.allclose(result, z)


# TODO : need onnx opset version at least 14 for uint8
@pytest.mark.xfail
def test_Sub_rand():
    x = np.random.randn(3, 4, 5).astype(np.float32)
    y = np.random.randn(3, 4, 5).astype(np.float32)
    z = x - y

    op = Sub(x, y)
    onnx_model = convert(OperationGraph([op]))

    results = onnxruntime.backend.run(onnx_model, [])
    assert len(results) == 1
    result = results[0]
    assert np.allclose(result, z)

    x = np.random.randint(12, 24, size=(3, 4, 5), dtype=np.uint8)
    y = np.random.randint(12, size=(3, 4, 5), dtype=np.uint8)
    z = x - y
    op = Sub(x, y)
    onnx_model = convert(OperationGraph([op]))

    results = onnxruntime.backend.run(onnx_model, [])
    assert len(results) == 1
    result = results[0]
    assert np.allclose(result, z)


def test_Sub_broadcast():
    x = np.random.randn(3, 4, 5).astype(np.float32)
    y = np.random.randn(5).astype(np.float32)
    z = x - y

    op = Sub(x, y)
    onnx_model = convert(OperationGraph([op]))

    results = onnxruntime.backend.run(onnx_model, [])
    assert len(results) == 1
    result = results[0]
    assert np.allclose(result, z)
