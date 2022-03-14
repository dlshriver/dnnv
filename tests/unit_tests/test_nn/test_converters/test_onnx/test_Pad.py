import numpy as np
import onnxruntime.backend

from dnnv.nn.converters.onnx import OnnxConverter
from dnnv.nn.graph import OperationGraph
from dnnv.nn.operations import *


def test_Pad():
    x = np.random.randn(1, 3, 4, 5).astype(np.float32)
    pads = np.array([0, 0, 1, 3, 0, 0, 2, 4]).astype(np.int64)
    value = np.float32(1.2)
    y = np.full((1, 3, 7, 12), value)
    y[:, :, 1:-2, 3:-4] = x

    op = Pad(x, pads, value=value)
    onnx_model = OnnxConverter.convert(OperationGraph([op]))
    result = onnxruntime.backend.run(onnx_model, [])[0]
    assert np.allclose(result, y)

    op = Pad(Input((1, 3, 4, 5), np.dtype(np.float32)), pads, value=value)
    onnx_model = OnnxConverter.convert(OperationGraph([op]))
    result = onnxruntime.backend.run(onnx_model, [x])[0]
    assert np.allclose(result, y)


def test_Pad_edge():
    x = np.random.randn(1, 3, 4, 5).astype(np.int32)
    pads = np.array([0, 0, 1, 1, 0, 0, 1, 1]).astype(np.int64)
    y = np.full((1, 3, 6, 7), 0)
    y[:, :, 1:-1, 1:-1] = x
    y[:, :, :1, :] = y[:, :, 1:2, :]
    y[:, :, -1:, :] = y[:, :, -2:-1, :]
    y[:, :, :, :1] = y[:, :, :, 1:2]
    y[:, :, :, -1:] = y[:, :, :, -2:-1]

    op = Pad(x, pads, mode="edge")
    onnx_model = OnnxConverter.convert(OperationGraph([op]))
    result = onnxruntime.backend.run(onnx_model, [])[0]
    assert np.allclose(result, y)


def test_Pad_reflect():
    x = np.random.randn(1, 3, 4, 5).astype(np.int32)
    pads = np.array([0, 0, 1, 1, 0, 0, 1, 1]).astype(np.int64)
    y = np.full((1, 3, 6, 7), 0)
    y[:, :, 1:-1, 1:-1] = x
    y[:, :, :1, :] = y[:, :, 2:3, :]
    y[:, :, -1:, :] = y[:, :, -3:-2, :]
    y[:, :, :, :1] = y[:, :, :, 2:3]
    y[:, :, :, -1:] = y[:, :, :, -3:-2]

    op = Pad(x, pads, mode="reflect")
    onnx_model = OnnxConverter.convert(OperationGraph([op]))
    result = onnxruntime.backend.run(onnx_model, [])[0]
    assert np.allclose(result, y)


def test_Pad_examples():
    data = np.array([[1.0, 1.2], [2.3, 3.4], [4.5, 5.7]])
    pads = np.array([0, 2, 0, 0], dtype=np.int64)
    output = np.array(
        [
            [0.0, 0.0, 1.0, 1.2],
            [0.0, 0.0, 2.3, 3.4],
            [0.0, 0.0, 4.5, 5.7],
        ]
    )
    op = Pad(data, pads, mode="constant")
    onnx_model = OnnxConverter.convert(OperationGraph([op]))
    result = onnxruntime.backend.run(onnx_model, [])[0]
    assert np.allclose(result, output)

    output = np.array(
        [
            [1.0, 1.0, 1.0, 1.2],
            [2.3, 2.3, 2.3, 3.4],
            [4.5, 4.5, 4.5, 5.7],
        ]
    )
    op = Pad(data, pads, mode="edge")
    onnx_model = OnnxConverter.convert(OperationGraph([op]))
    result = onnxruntime.backend.run(onnx_model, [])[0]
    assert np.allclose(result, output)

    pads = np.array([0, 1, 0, 0], dtype=np.int64)
    output = np.array(
        [
            [1.2, 1.0, 1.2],
            [3.4, 2.3, 3.4],
            [5.7, 4.5, 5.7],
        ]
    )
    op = Pad(data, pads, mode="reflect")
    onnx_model = OnnxConverter.convert(OperationGraph([op]))
    result = onnxruntime.backend.run(onnx_model, [])[0]
    assert np.allclose(result, output)
