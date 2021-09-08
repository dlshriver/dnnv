import numpy as np
import onnxruntime.backend

from dnnv.nn.converters.onnx import *
from dnnv.nn.operations import *


def test_Add_consts():
    op = Add(3, 4)
    onnx_model = convert(OperationGraph([op]))
    output = onnxruntime.backend.run(onnx_model, [])

    assert len(output) == 1
    assert output[0] == 7


def test_Add_a_is_op():
    op = Add(Input((-1, 5), np.dtype(np.float32)), 1.0)
    onnx_model = convert(OperationGraph([op]))

    x = np.array([[1.0, 2.0, 3.0, 4.0, 5.0]], dtype=np.float32)
    results = onnxruntime.backend.run(onnx_model, [x])
    assert len(results) == 1
    result = results[0]

    y = np.array([[2.0, 3.0, 4.0, 5.0, 6.0]], dtype=np.float32)
    assert np.all(result == y)


def test_Add_b_is_op():
    op = Add(0.5, Input((-1, 5), np.dtype(np.float32)))
    onnx_model = convert(OperationGraph([op]))

    x = np.array([[-1.0, -2.0, -3.0, -4.0, -5.0]], dtype=np.float32)
    results = onnxruntime.backend.run(onnx_model, [x])
    assert len(results) == 1
    result = results[0]

    y = np.array([[-0.5, -1.5, -2.5, -3.5, -4.5]], dtype=np.float32)
    assert np.all(result == y)


def test_Add_a_b_are_ops():
    op = Add(Input((-1, 5), np.dtype(np.float32)), Input((-1, 5), np.dtype(np.float32)))
    onnx_model = convert(OperationGraph([op]))

    a = np.array([[-1.0, -2.0, -3.0, -4.0, -5.0]], dtype=np.float32)
    b = np.array([[1.0, 2.0, 3.0, 4.0, 5.0]], dtype=np.float32)
    results = onnxruntime.backend.run(onnx_model, [a, b])
    assert len(results) == 1
    result = results[0]

    y = np.zeros((1, 5), dtype=np.float32)
    assert np.all(result == y)
