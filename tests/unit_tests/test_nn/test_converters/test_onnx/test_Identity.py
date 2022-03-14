import numpy as np
import onnxruntime.backend

from dnnv.nn.converters.onnx import OnnxConverter
from dnnv.nn.graph import OperationGraph
from dnnv.nn.operations import *


def test_Identity():
    data = np.array(
        [
            [
                [
                    [1, 2],
                    [3, 4],
                ]
            ]
        ],
        dtype=np.float32,
    )
    op = Identity(data)
    onnx_model = OnnxConverter.convert(OperationGraph([op]))
    result = onnxruntime.backend.run(onnx_model, [])[0]
    assert result.shape == data.shape
    assert np.allclose(result, data)

    input_op = Input((1, 1, 2, 2), np.dtype(np.float32))
    op = Identity(input_op)
    onnx_model = OnnxConverter.convert(OperationGraph([op]))
    result = onnxruntime.backend.run(onnx_model, [data])[0]
    assert result.shape == data.shape
    assert np.allclose(result, data)
