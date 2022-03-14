import numpy as np
import onnxruntime.backend

from dnnv.nn.converters.onnx import OnnxConverter
from dnnv.nn.graph import OperationGraph
from dnnv.nn.operations import *


def test_Unsqueeze():
    x = np.random.randn(1, 3, 1, 5).astype(np.float32)
    axes = np.array([0]).astype(np.int64)
    y = np.expand_dims(x, axis=0)

    op = Unsqueeze(x, axes)
    onnx_model = OnnxConverter.convert(OperationGraph([op]))
    result = onnxruntime.backend.run(onnx_model, [])[0]
    assert np.allclose(result, y)

    op = Unsqueeze(Input((1, 3, 1, 5), np.float32), axes)
    onnx_model = OnnxConverter.convert(OperationGraph([op]))
    result = onnxruntime.backend.run(onnx_model, [x])[0]
    assert np.allclose(result, y)


def test_Unsqueeze_negative_axes():
    x = np.random.randn(1, 3, 1, 5).astype(np.float32)
    axes = np.array([-2]).astype(np.int64)
    y = np.expand_dims(x, axis=-2)

    op = Unsqueeze(x, axes)
    onnx_model = OnnxConverter.convert(OperationGraph([op]))
    result = onnxruntime.backend.run(onnx_model, [])[0]
    assert np.allclose(result, y)


def test_Unsqueeze_one_axis():
    x = np.random.randn(3, 4, 5).astype(np.float32)

    for i in range(x.ndim):
        axes = np.array([i]).astype(np.int64)
        y = np.expand_dims(x, axis=i)

        op = Unsqueeze(x, axes)
        onnx_model = OnnxConverter.convert(OperationGraph([op]))
        result = onnxruntime.backend.run(onnx_model, [])[0]
        assert np.allclose(result, y)


def test_Unsqueeze_three_axes():
    x = np.random.randn(3, 4, 5).astype(np.float32)
    axes = np.array([2, 4, 5]).astype(np.int64)
    y = np.expand_dims(x, axis=2)
    y = np.expand_dims(y, axis=4)
    y = np.expand_dims(y, axis=5)

    op = Unsqueeze(x, axes)
    onnx_model = OnnxConverter.convert(OperationGraph([op]))
    result = onnxruntime.backend.run(onnx_model, [])[0]
    assert np.allclose(result, y)


def test_Unsqueeze_two_axes():
    x = np.random.randn(3, 4, 5).astype(np.float32)
    axes = np.array([1, 4]).astype(np.int64)
    y = np.expand_dims(x, axis=1)
    y = np.expand_dims(y, axis=4)

    op = Unsqueeze(x, axes)
    onnx_model = OnnxConverter.convert(OperationGraph([op]))
    result = onnxruntime.backend.run(onnx_model, [])[0]
    assert np.allclose(result, y)


def test_Unsqueeze_unsorted_axes():
    x = np.random.randn(3, 4, 5).astype(np.float32)
    axes = np.array([5, 4, 2]).astype(np.int64)
    y = np.expand_dims(x, axis=2)
    y = np.expand_dims(y, axis=4)
    y = np.expand_dims(y, axis=5)

    op = Unsqueeze(x, axes)
    onnx_model = OnnxConverter.convert(OperationGraph([op]))
    result = onnxruntime.backend.run(onnx_model, [])[0]
    assert np.allclose(result, y)
