import numpy as np
import onnxruntime.backend

from dnnv.nn.converters.onnx import *
from dnnv.nn.operations import *


def test_Clip():
    x = np.array([-2, 0, 2]).astype(np.float32)
    min_val = np.float32(-1)
    max_val = np.float32(1)
    y = np.clip(x, min_val, max_val)  # expected output [-1., 0., 1.]

    op = Clip(x, min=min_val, max=max_val)
    onnx_model = convert(OperationGraph([op]))
    output = onnxruntime.backend.run(onnx_model, [x, min_val, max_val])
    assert np.all(output == y)

    x = np.random.randn(3, 4, 5).astype(np.float32)
    y = np.clip(x, min_val, max_val)

    op = Clip(x, min=min_val, max=max_val)
    onnx_model = convert(OperationGraph([op]))
    output = onnxruntime.backend.run(onnx_model, [x, min_val, max_val])
    assert np.all(output == y)

    min_val = np.float32(-5)
    max_val = np.float32(5)
    x = np.array([-1, 0, 1]).astype(np.float32)
    y = np.array([-1, 0, 1]).astype(np.float32)

    op = Clip(x, min=min_val, max=max_val)
    onnx_model = convert(OperationGraph([op]))
    output = onnxruntime.backend.run(onnx_model, [x, min_val, max_val])
    assert np.all(output == y)

    x = np.array([-6, 0, 6]).astype(np.float32)
    y = np.array([-5, 0, 5]).astype(np.float32)

    op = Clip(x, min=min_val, max=max_val)
    onnx_model = convert(OperationGraph([op]))
    output = onnxruntime.backend.run(onnx_model, [x, min_val, max_val])
    assert np.all(output == y)

    x = np.array([-1, 0, 6]).astype(np.float32)
    y = np.array([-1, 0, 5]).astype(np.float32)

    op = Clip(x, min=min_val, max=max_val)
    onnx_model = convert(OperationGraph([op]))
    output = onnxruntime.backend.run(onnx_model, [x, min_val, max_val])
    assert np.all(output == y)


def test_Clip_Default():
    min_val = np.float32(0)
    max_val = np.inf
    x = np.random.randn(3, 4, 5).astype(np.float32)
    y = np.clip(x, min_val, max_val)

    op = Clip(x, min=min_val, max=max_val)
    onnx_model = convert(OperationGraph([op]))
    output = onnxruntime.backend.run(onnx_model, [x, min_val, max_val])
    assert np.all(output == y)

    min_val = -np.inf
    max_val = np.float32(0)
    x = np.random.randn(3, 4, 5).astype(np.float32)
    y = np.clip(x, min_val, max_val)

    op = Clip(x, min=min_val, max=max_val)
    onnx_model = convert(OperationGraph([op]))
    output = onnxruntime.backend.run(onnx_model, [x, min_val, max_val])
    assert np.all(output == y)

    min_val = None
    max_val = None
    x = np.array([-1, 0, 1]).astype(np.float32)
    y = np.array([-1, 0, 1]).astype(np.float32)

    op = Clip(x, min=min_val, max=max_val)
    onnx_model = convert(OperationGraph([op]))
    output = onnxruntime.backend.run(onnx_model, [x, min_val, max_val])
    assert np.all(output == y)


def test_Clip_Default_int8():
    min_val = np.int8(0)
    max_val = np.int8(np.iinfo(np.int8).max)
    x = np.random.randn(3, 4, 5).astype(np.int8)
    y = np.clip(x, min_val, max_val)

    op = Clip(x, min=min_val, max=max_val)
    onnx_model = convert(OperationGraph([op]))
    output = onnxruntime.backend.run(onnx_model, [x, min_val, max_val])
    assert np.all(output == y)

    min_val = np.int8(np.iinfo(np.int8).min)
    max_val = np.int8(0)
    x = np.random.randn(3, 4, 5).astype(np.int8)
    y = np.clip(x, min_val, max_val)

    op = Clip(x, min=min_val, max=max_val)
    onnx_model = convert(OperationGraph([op]))
    output = onnxruntime.backend.run(onnx_model, [x, min_val, max_val])
    assert np.all(output == y)
