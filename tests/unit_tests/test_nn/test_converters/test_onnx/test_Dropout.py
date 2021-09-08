import numpy as np
import onnxruntime
import pytest

from dnnv.nn.converters.onnx import *
from dnnv.nn.operations import *


def test_Dropout_consts():
    x = np.array([3, 4]).astype(np.float32)
    op = Dropout(x)
    onnx_model = convert(OperationGraph([OutputSelect(op, 0)]))

    results = onnxruntime.backend.run(onnx_model, [])
    assert len(results) == 1
    result = results[0]

    y = np.array([3, 4]).astype(np.float32)
    assert np.allclose(result, y)


def test_Dropout_x_is_op():
    x = np.array([3, 4]).astype(np.float32)
    input_op = Input((2,), np.dtype(np.float32))
    op = Dropout(input_op)
    onnx_model = convert(OperationGraph([OutputSelect(op, 0)]))

    results = onnxruntime.backend.run(onnx_model, [x])
    assert len(results) == 1
    result = results[0]

    y = np.array([3, 4]).astype(np.float32)
    assert np.allclose(result, y)
