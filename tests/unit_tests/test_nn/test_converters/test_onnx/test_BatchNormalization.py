import numpy as np
import onnxruntime

from dnnv.nn.converters.onnx import *
from dnnv.nn.operations import *


def test_BatchNormalization_consts():
    x = np.arange(12).astype(np.float32).reshape((1, 3, 2, 2))
    scale = np.full(3, 2.0, dtype=np.float32)
    bias = np.full(3, 0.0, dtype=np.float32)
    mean = np.full(3, 5.5, dtype=np.float32)
    var = np.full(3, 11.9, dtype=np.float32)

    op = BatchNormalization(x, scale, bias, mean, var)
    onnx_model = convert(OperationGraph([op]))

    results = onnxruntime.backend.run(onnx_model, [])
    assert len(results) == 1
    result = results[0]

    y = np.array(
        [
            [
                [[-3.1887393, -2.6089685], [-2.0291977, -1.4494269]],
                [[-0.8696561, -0.28988528], [0.28988552, 0.8696561]],
                [[1.4494271, 2.0291982], [2.6089687, 3.1887393]],
            ]
        ],
        dtype=np.float32,
    )
    assert np.allclose(result, y)


def test_BatchNormalization_x_is_op():
    x = np.arange(12).astype(np.float32).reshape((1, 3, 2, 2))
    scale = np.full(3, 2.0, dtype=np.float32)
    bias = np.full(3, 0.0, dtype=np.float32)
    mean = np.full(3, 5.5, dtype=np.float32)
    var = np.full(3, 11.9, dtype=np.float32)

    input_op = Input((1, 3, 2, 2), np.dtype(np.float32))
    op = BatchNormalization(input_op, scale, bias, mean, var)
    onnx_model = convert(OperationGraph([op]))

    results = onnxruntime.backend.run(onnx_model, [x])
    assert len(results) == 1
    result = results[0]

    y = np.array(
        [
            [
                [[-3.1887393, -2.6089685], [-2.0291977, -1.4494269]],
                [[-0.8696561, -0.28988528], [0.28988552, 0.8696561]],
                [[1.4494271, 2.0291982], [2.6089687, 3.1887393]],
            ]
        ],
        dtype=np.float32,
    )
    assert np.allclose(result, y)
