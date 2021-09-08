import numpy as np
import onnxruntime

from dnnv.nn.converters.onnx import *
from dnnv.nn.operations import *


def test_MatMul_2d():
    a = np.random.randn(3, 4).astype(np.float32)
    b = np.random.randn(4, 3).astype(np.float32)
    op = MatMul(a, b)
    onnx_model = convert(OperationGraph([op]))

    results = onnxruntime.backend.run(onnx_model, [])
    assert len(results) == 1
    result = results[0]

    y = a @ b
    assert np.allclose(result, y)

    op = MatMul(
        Input((3, 4), np.dtype(np.float32)), Input((4, 3), np.dtype(np.float32))
    )
    onnx_model = convert(OperationGraph([op]))

    results = onnxruntime.backend.run(onnx_model, [a, b])
    assert len(results) == 1
    result = results[0]

    assert np.allclose(result, y)


def test_MatMul_3d():
    a = np.random.randn(2, 3, 4).astype(np.float32)
    b = np.random.randn(2, 4, 3).astype(np.float32)
    op = MatMul(a, b)
    onnx_model = convert(OperationGraph([op]))

    results = onnxruntime.backend.run(onnx_model, [])
    assert len(results) == 1
    result = results[0]

    y = a @ b
    assert np.allclose(result, y)


def test_MatMul_4d():
    a = np.random.randn(1, 2, 3, 4).astype(np.float32)
    b = np.random.randn(1, 2, 4, 3).astype(np.float32)
    op = MatMul(a, b)
    onnx_model = convert(OperationGraph([op]))

    results = onnxruntime.backend.run(onnx_model, [])
    assert len(results) == 1
    result = results[0]

    y = a @ b
    assert np.allclose(result, y)


def test_MatMul_mixedD():
    a = np.random.randn(3, 4).astype(np.float32)
    b = np.random.randn(
        4,
    ).astype(np.float32)
    op = MatMul(a, b)
    onnx_model = convert(OperationGraph([op]))

    results = onnxruntime.backend.run(onnx_model, [])
    assert len(results) == 1
    result = results[0]

    y = a @ b
    assert np.allclose(result, y)

    a = np.random.randn(
        4,
    ).astype(np.float32)
    b = np.random.randn(4, 3).astype(np.float32)
    op = MatMul(a, b)
    onnx_model = convert(OperationGraph([op]))

    results = onnxruntime.backend.run(onnx_model, [])
    assert len(results) == 1
    result = results[0]

    y = a @ b
    assert np.allclose(result, y)
