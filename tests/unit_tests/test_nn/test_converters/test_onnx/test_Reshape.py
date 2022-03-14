import numpy as np
import onnxruntime.backend

from dnnv.nn.converters.onnx import OnnxConverter
from dnnv.nn.graph import OperationGraph
from dnnv.nn.operations import *


def test_Reshape():
    original_shape = [0, 3, 4]
    data = np.random.random_sample(original_shape).astype(np.float32)
    new_shape = np.array([3, 4, 0], dtype=np.int64)
    y = np.reshape(data, new_shape)

    op = Reshape(data, new_shape, allowzero=True)
    onnx_model = OnnxConverter.convert(OperationGraph([op]), opset=14)

    results = onnxruntime.backend.run(onnx_model, [])
    assert len(results) == 1
    result = results[0]
    assert np.allclose(result, y)

    op = Reshape(
        Input((0, 3, 4), np.dtype(np.float32)),
        new_shape,
        allowzero=True,
    )
    onnx_model = OnnxConverter.convert(OperationGraph([op]), ir_version=8, opset=14)

    results = onnxruntime.backend.run(onnx_model, [data, new_shape])
    assert len(results) == 1
    result = results[0]
    assert np.allclose(result, y)


def test_Reshape_reordered_all_dims():
    original_shape = [2, 3, 4]
    data = np.random.random_sample(original_shape).astype(np.float32)
    new_shape = np.array([4, 2, 3], dtype=np.int64)
    y = np.reshape(data, new_shape)

    op = Reshape(data, new_shape)
    onnx_model = OnnxConverter().convert(OperationGraph([op]))

    results = onnxruntime.backend.run(onnx_model, [])
    assert len(results) == 1
    result = results[0]
    assert np.allclose(result, y)


def test_Reshape_reordered_last_dims():
    original_shape = [2, 3, 4]
    data = np.random.random_sample(original_shape).astype(np.float32)
    new_shape = np.array([2, 4, 3], dtype=np.int64)
    y = np.reshape(data, new_shape)

    op = Reshape(data, new_shape)
    onnx_model = OnnxConverter().convert(OperationGraph([op]))

    results = onnxruntime.backend.run(onnx_model, [])
    assert len(results) == 1
    result = results[0]
    assert np.allclose(result, y)


def test_Reshape_reduced_dims():
    original_shape = [2, 3, 4]
    data = np.random.random_sample(original_shape).astype(np.float32)
    new_shape = np.array([2, 12], dtype=np.int64)
    y = np.reshape(data, new_shape)

    op = Reshape(data, new_shape)
    onnx_model = OnnxConverter().convert(OperationGraph([op]))

    results = onnxruntime.backend.run(onnx_model, [])
    assert len(results) == 1
    result = results[0]
    assert np.allclose(result, y)


def test_Reshape_extended_dims():
    original_shape = [2, 3, 4]
    data = np.random.random_sample(original_shape).astype(np.float32)
    new_shape = np.array([2, 3, 2, 2], dtype=np.int64)
    y = np.reshape(data, new_shape)

    op = Reshape(data, new_shape)
    onnx_model = OnnxConverter().convert(OperationGraph([op]))

    results = onnxruntime.backend.run(onnx_model, [])
    assert len(results) == 1
    result = results[0]
    assert np.allclose(result, y)


def test_Reshape_one_dim():
    original_shape = [2, 3, 4]
    data = np.random.random_sample(original_shape).astype(np.float32)
    new_shape = np.array([24], dtype=np.int64)
    y = np.reshape(data, new_shape)

    op = Reshape(data, new_shape)
    onnx_model = OnnxConverter().convert(OperationGraph([op]))

    results = onnxruntime.backend.run(onnx_model, [])
    assert len(results) == 1
    result = results[0]
    assert np.allclose(result, y)


def test_Reshape_negative_dim():
    original_shape = [2, 3, 4]
    data = np.random.random_sample(original_shape).astype(np.float32)
    new_shape = np.array([2, -1, 2], dtype=np.int64)
    y = np.reshape(data, new_shape)

    op = Reshape(data, new_shape)
    onnx_model = OnnxConverter().convert(OperationGraph([op]))

    results = onnxruntime.backend.run(onnx_model, [])
    assert len(results) == 1
    result = results[0]
    assert np.allclose(result, y)


def test_Reshape_negative_extended_dims():
    original_shape = [2, 3, 4]
    data = np.random.random_sample(original_shape).astype(np.float32)
    new_shape = np.array([-1, 2, 3, 4], dtype=np.int64)
    y = np.reshape(data, new_shape)

    op = Reshape(data, new_shape)
    onnx_model = OnnxConverter().convert(OperationGraph([op]))

    results = onnxruntime.backend.run(onnx_model, [])
    assert len(results) == 1
    result = results[0]
    assert np.allclose(result, y)


def test_Reshape_zero_dim():
    original_shape = [2, 3, 4]
    data = np.random.random_sample(original_shape).astype(np.float32)
    new_shape = np.array([2, 0, 4, 1], dtype=np.int64)
    y = np.reshape(data, [2, 3, 4, 1])

    op = Reshape(data, new_shape)
    onnx_model = OnnxConverter().convert(OperationGraph([op]))

    results = onnxruntime.backend.run(onnx_model, [])
    assert len(results) == 1
    result = results[0]
    assert np.allclose(result, y)


def test_Reshape_zero_and_negative_dim():
    original_shape = [2, 3, 4]
    data = np.random.random_sample(original_shape).astype(np.float32)
    new_shape = np.array([2, 0, 1, -1], dtype=np.int64)
    y = np.reshape(data, [2, 3, 1, -1])

    op = Reshape(data, new_shape)
    onnx_model = OnnxConverter().convert(OperationGraph([op]))

    results = onnxruntime.backend.run(onnx_model, [])
    assert len(results) == 1
    result = results[0]
    assert np.allclose(result, y)
