import numpy as np
import onnxruntime.backend

from dnnv.nn.converters.onnx import OnnxConverter
from dnnv.nn.graph import OperationGraph
from dnnv.nn.operations import *


def test_Softmax():
    x = np.array([[-1, 0, 1]]).astype(np.float32)
    y = np.array([[0.09003058, 0.24472848, 0.66524094]])

    op = Softmax(x, axis=1)
    onnx_model = OnnxConverter.convert(OperationGraph([op]))

    results = onnxruntime.backend.run(onnx_model, [])
    assert len(results) == 1
    result = results[0]
    assert np.allclose(result, y)


def test_Softmax_axis_default():
    x = np.array([[0, 1, 2, 3], [10000, 10001, 10002, 10003]]).astype(np.float32)
    y = np.array(
        [
            [0.032058604, 0.08714432, 0.23688284, 0.6439143],
            [0.032058604, 0.08714432, 0.23688284, 0.6439143],
        ]
    )

    op = Softmax(x)
    onnx_model = OnnxConverter.convert(OperationGraph([op]))

    results = onnxruntime.backend.run(onnx_model, [])
    assert len(results) == 1
    result = results[0]
    assert np.allclose(result, y)
