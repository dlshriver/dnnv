import numpy as np
import onnxruntime
import pytest

from dnnv.nn.converters.onnx import *
from dnnv.nn.operations import *


def test_MaxPool_1d_default():
    x = np.random.randn(1, 3, 32).astype(np.float32)
    y = np.empty((1, 3, 31), dtype=np.float32)
    for idx in np.ndindex(y.shape):
        y[idx] = max(x[idx], x[idx[:-1] + (idx[-1] + 1,)])

    op = MaxPool(x, np.array([2]))
    onnx_model = convert(OperationGraph([op]))

    results = onnxruntime.backend.run(onnx_model, [])
    assert len(results) == 1
    result = results[0]
    assert np.allclose(result, y)

    op = MaxPool(Input((1, 3, 32), np.dtype(np.float32)), np.array([2]))
    onnx_model = convert(OperationGraph([op]))

    results = onnxruntime.backend.run(onnx_model, [x])
    assert len(results) == 1
    result = results[0]
    assert np.allclose(result, y)


@pytest.mark.xfail
def test_MaxPool_2d_ceil():
    x = np.array(
        [
            [
                [
                    [1, 2, 3, 4],
                    [5, 6, 7, 8],
                    [9, 10, 11, 12],
                    [13, 14, 15, 16],
                ]
            ]
        ]
    ).astype(np.float32)

    op = MaxPool(x, np.array([3, 3]), strides=np.array([2, 2]), ceil_mode=True)
    onnx_model = convert(OperationGraph([op]))

    results = onnxruntime.backend.run(onnx_model, [])
    assert len(results) == 1
    result = results[0]

    y = np.array([[[[11, 12], [15, 16]]]]).astype(np.float32)
    assert np.allclose(result, y)


def test_MaxPool_2d_default():
    x = np.random.randn(1, 3, 32, 32).astype(np.float32)
    y = np.empty((1, 3, 31, 31), dtype=np.float32)
    for idx in np.ndindex(y.shape):
        n, c, h, w = idx
        y[idx] = max(
            x[idx], x[n, c, h, w + 1], x[n, c, h + 1, w], x[n, c, h + 1, w + 1]
        )

    op = MaxPool(x, np.array([2, 2]))
    onnx_model = convert(OperationGraph([op]))

    results = onnxruntime.backend.run(onnx_model, [])
    assert len(results) == 1
    result = results[0]
    assert np.allclose(result, y)


def test_MaxPool_2d_dilations():
    x = np.array(
        [
            [
                [
                    [1, 2, 3, 4],
                    [5, 6, 7, 8],
                    [9, 10, 11, 12],
                    [13, 14, 15, 16],
                ]
            ]
        ]
    ).astype(np.float32)
    y = np.array([[[[11, 12], [15, 16]]]]).astype(np.float32)

    op = MaxPool(
        x, np.array([2, 2]), strides=np.array([1, 1]), dilations=np.array([2, 2])
    )
    onnx_model = convert(OperationGraph([op]))

    results = onnxruntime.backend.run(onnx_model, [])
    assert len(results) == 1
    result = results[0]
    assert np.allclose(result, y)


def test_MaxPool_2d_pads():
    x = np.random.randn(1, 3, 28, 28).astype(np.float32)
    y = np.full((1, 3, 30, 30), -np.inf, dtype=np.float32)
    for idx in np.ndindex(y.shape):
        n, c, h, w = idx
        for i in range(3):
            for j in range(3):
                if h - 2 + i < 0:
                    continue
                if w - 2 + j < 0:
                    continue
                if h - 2 + i >= 28:
                    continue
                if w - 2 + j >= 28:
                    continue
                y[idx] = max(y[idx], x[n, c, h - 2 + i, w - 2 + j])

    op = MaxPool(x, np.array([3, 3]), pads=np.array([2, 2, 2, 2]))
    onnx_model = convert(OperationGraph([op]))

    results = onnxruntime.backend.run(onnx_model, [])
    assert len(results) == 1
    result = results[0]
    assert np.allclose(result, y)


def test_MaxPool_2d_precomputed_pads():
    x = np.array(
        [
            [
                [
                    [1, 2, 3, 4, 5],
                    [6, 7, 8, 9, 10],
                    [11, 12, 13, 14, 15],
                    [16, 17, 18, 19, 20],
                    [21, 22, 23, 24, 25],
                ]
            ]
        ]
    ).astype(np.float32)
    y = np.array(
        [
            [
                [
                    [13, 14, 15, 15, 15],
                    [18, 19, 20, 20, 20],
                    [23, 24, 25, 25, 25],
                    [23, 24, 25, 25, 25],
                    [23, 24, 25, 25, 25],
                ]
            ]
        ]
    ).astype(np.float32)

    op = MaxPool(x, np.array([5, 5]), pads=np.array([2, 2, 2, 2]))
    onnx_model = convert(OperationGraph([op]))

    results = onnxruntime.backend.run(onnx_model, [])
    assert len(results) == 1
    result = results[0]
    assert np.allclose(result, y)


def test_MaxPool_2d_precomputed_strides():
    x = np.array(
        [
            [
                [
                    [1, 2, 3, 4, 5],
                    [6, 7, 8, 9, 10],
                    [11, 12, 13, 14, 15],
                    [16, 17, 18, 19, 20],
                    [21, 22, 23, 24, 25],
                ]
            ]
        ]
    ).astype(np.float32)
    y = np.array([[[[7, 9], [17, 19]]]]).astype(np.float32)

    op = MaxPool(x, np.array([2, 2]), strides=np.array([2, 2]))
    onnx_model = convert(OperationGraph([op]))

    results = onnxruntime.backend.run(onnx_model, [])
    assert len(results) == 1
    result = results[0]
    assert np.allclose(result, y)


def test_MaxPool_2d_strides():
    x = np.random.randn(1, 3, 32, 32).astype(np.float32)
    y = np.full((1, 3, 10, 10), -np.inf, dtype=np.float32)
    for idx in np.ndindex(y.shape):
        n, c, h, w = idx
        for i in range(5):
            for j in range(5):
                y[idx] = max(y[idx], x[n, c, 3 * h + i, 3 * w + j])

    op = MaxPool(x, np.array([5, 5]), strides=np.array([3, 3]))
    onnx_model = convert(OperationGraph([op]))

    results = onnxruntime.backend.run(onnx_model, [])
    assert len(results) == 1
    result = results[0]
    assert np.allclose(result, y)


def test_MaxPool_2d_uint8():
    x = np.array(
        [
            [
                [
                    [1, 2, 3, 4, 5],
                    [6, 7, 8, 9, 10],
                    [11, 12, 13, 14, 15],
                    [16, 17, 18, 19, 20],
                    [21, 22, 23, 24, 25],
                ]
            ]
        ]
    ).astype(np.uint8)
    y = np.array(
        [
            [
                [
                    [13, 14, 15, 15, 15],
                    [18, 19, 20, 20, 20],
                    [23, 24, 25, 25, 25],
                    [23, 24, 25, 25, 25],
                    [23, 24, 25, 25, 25],
                ]
            ]
        ]
    ).astype(np.uint8)

    op = MaxPool(x, np.array([5, 5]), pads=np.array([2, 2, 2, 2]))
    onnx_model = convert(OperationGraph([op]))

    results = onnxruntime.backend.run(onnx_model, [])
    assert len(results) == 1
    result = results[0]
    assert np.allclose(result, y)


def test_MaxPool_3d_default():
    x = np.random.randn(1, 3, 32, 32, 32).astype(np.float32)
    y = np.full((1, 3, 31, 31, 31), -np.inf, dtype=np.float32)
    for idx in np.ndindex(y.shape):
        n, c, h, w, d = idx
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    y[idx] = max(y[idx], x[n, c, h + i, w + j, d + k])

    op = MaxPool(x, np.array([2, 2, 2]))
    onnx_model = convert(OperationGraph([op]))

    results = onnxruntime.backend.run(onnx_model, [])
    assert len(results) == 1
    result = results[0]
    assert np.allclose(result, y)
