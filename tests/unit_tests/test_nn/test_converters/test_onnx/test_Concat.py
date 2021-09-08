import numpy as np
import onnxruntime

from dnnv.nn.converters.onnx import *
from dnnv.nn.operations import *


def test_Concat_consts():
    x0 = np.arange(5)
    x1 = np.arange(10, 20)

    op = Concat([x0, x1], 0)
    onnx_model = convert(OperationGraph([op]))

    results = onnxruntime.backend.run(onnx_model, [])
    assert len(results) == 1
    result = results[0]

    y = np.array([0, 1, 2, 3, 4, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19])
    assert np.all(result == y)


def test_Concat_x0_is_op():
    x0 = np.arange(5)
    x1 = np.arange(10, 20)

    input_op0 = Input((5,), np.dtype(np.int64))
    op = Concat([input_op0, x1], 0)
    onnx_model = convert(OperationGraph([op]))

    results = onnxruntime.backend.run(onnx_model, [x0])
    assert len(results) == 1
    result = results[0]

    y = np.array([0, 1, 2, 3, 4, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19])
    assert np.all(result == y)


def test_Concat_x1_is_op():
    x0 = np.arange(5)
    x1 = np.arange(10, 20)

    input_op1 = Input((10,), np.dtype(np.int64))
    op = Concat([x0, input_op1], 0)
    onnx_model = convert(OperationGraph([op]))

    results = onnxruntime.backend.run(onnx_model, [x1])
    assert len(results) == 1
    result = results[0]

    y = np.array([0, 1, 2, 3, 4, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19])
    assert np.all(result == y)


def test_Concat_x0_x1_are_op():
    x0 = np.arange(5)
    x1 = np.arange(10, 20)

    input_op0 = Input((5,), np.dtype(np.int64))
    input_op1 = Input((10,), np.dtype(np.int64))
    op = Concat([input_op0, input_op1], 0)
    onnx_model = convert(OperationGraph([op]))

    results = onnxruntime.backend.run(onnx_model, [x0, x1])
    assert len(results) == 1
    result = results[0]

    y = np.array([0, 1, 2, 3, 4, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19])
    assert np.all(result == y)


def test_Concat_1d():
    x0 = np.array([1, 2], dtype=np.float32)
    x1 = np.array([3, 4], dtype=np.float32)

    op = Concat([x0, x1], 0)
    onnx_model = convert(OperationGraph([op]))

    results = onnxruntime.backend.run(onnx_model, [])
    assert len(results) == 1
    result = results[0]

    y = np.array([1, 2, 3, 4], dtype=np.float32)
    assert np.all(result == y)

    op = Concat([x0, x1], -1)
    onnx_model = convert(OperationGraph([op]))

    results = onnxruntime.backend.run(onnx_model, [])
    assert len(results) == 1
    result = results[0]

    y = np.array([1, 2, 3, 4], dtype=np.float32)
    assert np.all(result == y)


def test_Concat_2d():
    x0 = np.array([[1, 2], [3, 4]], dtype=np.float32)
    x1 = np.array([[5, 6], [7, 8]], dtype=np.float32)

    op = Concat([x0, x1], 0)
    onnx_model = convert(OperationGraph([op]))

    results = onnxruntime.backend.run(onnx_model, [])
    assert len(results) == 1
    result = results[0]

    y = np.array([[1, 2], [3, 4], [5, 6], [7, 8]], dtype=np.float32)
    assert np.all(result == y)

    op = Concat([x0, x1], 1)
    onnx_model = convert(OperationGraph([op]))

    results = onnxruntime.backend.run(onnx_model, [])
    assert len(results) == 1
    result = results[0]

    y = np.array([[1, 2, 5, 6], [3, 4, 7, 8]], dtype=np.float32)
    assert np.all(result == y)

    op = Concat([x0, x1], -1)
    onnx_model = convert(OperationGraph([op]))

    results = onnxruntime.backend.run(onnx_model, [])
    assert len(results) == 1
    result = results[0]

    y = np.array([[1, 2, 5, 6], [3, 4, 7, 8]], dtype=np.float32)
    assert np.all(result == y)

    op = Concat([x0, x1], -2)
    onnx_model = convert(OperationGraph([op]))

    results = onnxruntime.backend.run(onnx_model, [])
    assert len(results) == 1
    result = results[0]

    y = np.array([[1, 2], [3, 4], [5, 6], [7, 8]], dtype=np.float32)
    assert np.all(result == y)


def test_Concat_3d():
    x0 = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.float32)
    x1 = np.array([[[9, 10], [11, 12]], [[13, 14], [15, 16]]], dtype=np.float32)

    op = Concat([x0, x1], 0)
    onnx_model = convert(OperationGraph([op]))

    results = onnxruntime.backend.run(onnx_model, [])
    assert len(results) == 1
    result = results[0]

    y = np.array(
        [[[1, 2], [3, 4]], [[5, 6], [7, 8]], [[9, 10], [11, 12]], [[13, 14], [15, 16]]],
        dtype=np.float32,
    )
    assert np.all(result == y)

    op = Concat([x0, x1], 1)
    onnx_model = convert(OperationGraph([op]))

    results = onnxruntime.backend.run(onnx_model, [])
    assert len(results) == 1
    result = results[0]

    y = np.array(
        [[[1, 2], [3, 4], [9, 10], [11, 12]], [[5, 6], [7, 8], [13, 14], [15, 16]]],
        dtype=np.float32,
    )
    assert np.all(result == y)

    op = Concat([x0, x1], 2)
    onnx_model = convert(OperationGraph([op]))

    results = onnxruntime.backend.run(onnx_model, [])
    assert len(results) == 1
    result = results[0]

    y = np.array(
        [[[1, 2, 9, 10], [3, 4, 11, 12]], [[5, 6, 13, 14], [7, 8, 15, 16]]],
        dtype=np.float32,
    )
    assert np.all(result == y)

    op = Concat([x0, x1], -1)
    onnx_model = convert(OperationGraph([op]))

    results = onnxruntime.backend.run(onnx_model, [])
    assert len(results) == 1
    result = results[0]

    y = np.array(
        [[[1, 2, 9, 10], [3, 4, 11, 12]], [[5, 6, 13, 14], [7, 8, 15, 16]]],
        dtype=np.float32,
    )
    assert np.all(result == y)

    op = Concat([x0, x1], -2)
    onnx_model = convert(OperationGraph([op]))

    results = onnxruntime.backend.run(onnx_model, [])
    assert len(results) == 1
    result = results[0]

    y = np.array(
        [[[1, 2], [3, 4], [9, 10], [11, 12]], [[5, 6], [7, 8], [13, 14], [15, 16]]],
        dtype=np.float32,
    )
    assert np.all(result == y)

    op = Concat([x0, x1], -3)
    onnx_model = convert(OperationGraph([op]))

    results = onnxruntime.backend.run(onnx_model, [])
    assert len(results) == 1
    result = results[0]

    y = np.array(
        [[[1, 2], [3, 4]], [[5, 6], [7, 8]], [[9, 10], [11, 12]], [[13, 14], [15, 16]]],
        dtype=np.float32,
    )
    assert np.all(result == y)
