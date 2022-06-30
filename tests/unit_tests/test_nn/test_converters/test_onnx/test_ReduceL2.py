import numpy as np
import onnxruntime.backend
import pytest

from dnnv.nn.converters.onnx import *
from dnnv.nn.operations import *


def test_ReduceL2():
    shape = [3, 2, 2]
    axes = None
    keepdims = 1

    data = np.reshape(np.arange(1, np.prod(shape) + 1, dtype=np.float32), shape)
    # [[[1., 2.], [3., 4.]], [[5., 6.], [7., 8.]], [[9., 10.], [11., 12.]]]
    reduced = np.sqrt(np.sum(a=np.square(data), axis=axes, keepdims=keepdims == 1))
    # [[[25.49509757]]]

    op = ReduceL2(data, axes=axes, keepdims=keepdims)
    onnx_model = convert(OperationGraph([op]))
    output = onnxruntime.backend.run(onnx_model, [data])
    assert np.all(output == reduced)

    np.random.seed(0)
    data = np.random.uniform(-10, 10, shape).astype(np.float32)
    reduced = np.sqrt(np.sum(a=np.square(data), axis=axes, keepdims=keepdims == 1))

    op = ReduceL2(data, axes=axes, keepdims=keepdims)
    onnx_model = convert(OperationGraph([op]))
    output = onnxruntime.backend.run(onnx_model, [data])
    assert np.all(output == reduced)


@pytest.mark.xfail
def test_ReduceL2_do_not_keep_dims():
    shape = [3, 2, 2]
    axes = [2]
    keepdims = 0

    data = np.reshape(np.arange(1, np.prod(shape) + 1, dtype=np.float32), shape)
    # [[[1., 2.], [3., 4.]], [[5., 6.], [7., 8.]], [[9., 10.], [11., 12.]]]
    reduced = np.sqrt(
        np.sum(a=np.square(data), axis=tuple(axes), keepdims=keepdims == 1)
    )
    # [[2.23606798, 5.],
    # [7.81024968, 10.63014581],
    # [13.45362405, 16.2788206]]

    op = ReduceL2(data, axes=axes, keepdims=keepdims)
    onnx_model = convert(OperationGraph([op]))
    output = onnxruntime.backend.run(onnx_model, [data])
    assert np.all(output == reduced)

    np.random.seed(0)
    data = np.random.uniform(-10, 10, shape).astype(np.float32)
    reduced = np.sqrt(
        np.sum(a=np.square(data), axis=tuple(axes), keepdims=keepdims == 1)
    )

    op = ReduceL2(data, axes=axes, keepdims=keepdims)
    onnx_model = convert(OperationGraph([op]))
    output = onnxruntime.backend.run(onnx_model, [data])
    assert np.all(output == reduced)


@pytest.mark.xfail
def test_ReduceL2_keep_dims():
    shape = [3, 2, 2]
    axes = [2]
    keepdims = 1

    data = np.reshape(np.arange(1, np.prod(shape) + 1, dtype=np.float32), shape)
    # print(data)
    # [[[1., 2.], [3., 4.]], [[5., 6.], [7., 8.]], [[9., 10.], [11., 12.]]]
    reduced = np.sqrt(
        np.sum(a=np.square(data), axis=tuple(axes), keepdims=keepdims == 1)
    )
    # [[[2.23606798], [5.]]
    # [[7.81024968], [10.63014581]]
    # [[13.45362405], [16.2788206 ]]]

    op = ReduceL2(data, axes=axes, keepdims=keepdims)
    onnx_model = convert(OperationGraph([op]))
    output = onnxruntime.backend.run(onnx_model, [data])
    assert np.all(output == reduced)

    np.random.seed(0)
    data = np.random.uniform(-10, 10, shape).astype(np.float32)
    reduced = np.sqrt(
        np.sum(a=np.square(data), axis=tuple(axes), keepdims=keepdims == 1)
    )

    op = ReduceL2(data, axes=axes, keepdims=keepdims)
    onnx_model = convert(OperationGraph([op]))
    output = onnxruntime.backend.run(onnx_model, [data])
    assert np.all(output == reduced)


@pytest.mark.xfail
def test_ReduceL2_negative_axes_keepdims():
    shape = [3, 2, 2]
    axes = [-1]
    keepdims = 1

    data = np.reshape(np.arange(1, np.prod(shape) + 1, dtype=np.float32), shape)
    # [[[1., 2.], [3., 4.]], [[5., 6.], [7., 8.]], [[9., 10.], [11., 12.]]]
    reduced = np.sqrt(
        np.sum(a=np.square(data), axis=tuple(axes), keepdims=keepdims == 1)
    )
    # [[[2.23606798], [5.]]
    # [[7.81024968], [10.63014581]]
    # [[13.45362405], [16.2788206 ]]]

    op = ReduceL2(data, axes=axes, keepdims=keepdims)
    onnx_model = convert(OperationGraph([op]))
    output = onnxruntime.backend.run(onnx_model, [data])
    assert np.all(output == reduced)

    np.random.seed(0)
    data = np.random.uniform(-10, 10, shape).astype(np.float32)
    reduced = np.sqrt(
        np.sum(a=np.square(data), axis=tuple(axes), keepdims=keepdims == 1)
    )

    op = ReduceL2(data, axes=axes, keepdims=keepdims)
    onnx_model = convert(OperationGraph([op]))
    output = onnxruntime.backend.run(onnx_model, [data])
    assert np.all(output == reduced)
