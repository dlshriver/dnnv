import numpy as np
import onnxruntime

from dnnv.nn.converters.onnx import *
from dnnv.nn.operations import *


def test_Div_consts():
    a = np.array([3, 4]).astype(np.float32)
    b = np.array([1, 2]).astype(np.float32)
    op = Div(a, b)
    onnx_model = convert(OperationGraph([op]))

    results = onnxruntime.backend.run(onnx_model, [])
    assert len(results) == 1
    result = results[0]

    y = np.array([3, 2]).astype(np.float32)
    assert np.allclose(result, y)


def test_Div_a_is_op():
    op = Div(Input((-1, 5), np.dtype(np.float32)), 2.0)
    onnx_model = convert(OperationGraph([op]))

    x = np.array([[1.0, 2.0, 3.0, 4.0, 5.0]], dtype=np.float32)
    results = onnxruntime.backend.run(onnx_model, [x])
    assert len(results) == 1
    result = results[0]

    y = np.array([[0.5, 1.0, 1.5, 2.0, 2.5]], dtype=np.float32)
    assert np.allclose(result, y)


def test_Div_b_is_op():
    op = Div(10.0, Input((-1, 5), np.dtype(np.float32)))
    onnx_model = convert(OperationGraph([op]))

    x = np.array([[-1.0, -2.0, -3.0, -4.0, -5.0]], dtype=np.float32)
    results = onnxruntime.backend.run(onnx_model, [x])
    assert len(results) == 1
    result = results[0]

    y = np.array([[-10.0, -5.0, -3.3333333333, -2.5, -2]], dtype=np.float32)
    assert np.allclose(result, y)


def test_Div_a_b_are_ops():
    op = Div(Input((-1, 5), np.dtype(np.float32)), Input((-1, 5), np.dtype(np.float32)))
    onnx_model = convert(OperationGraph([op]))

    a = np.array([[-1.0, -2.0, -3.0, -4.0, -5.0]], dtype=np.float32)
    b = np.array([[1.0, 2.0, 3.0, 4.0, 5.0]], dtype=np.float32)
    results = onnxruntime.backend.run(onnx_model, [a, b])
    assert len(results) == 1
    result = results[0]

    y = -np.ones((1, 5), dtype=np.float32)
    assert np.allclose(result, y)
