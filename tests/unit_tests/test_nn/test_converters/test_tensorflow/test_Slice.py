import numpy as np
import onnxruntime.backend

from dnnv.nn.converters.tensorflow import *
from dnnv.nn.operations import *


def check(op, inputs, outputs):
    tf_op = TensorflowConverter().visit(op)
    result = tf_op(*inputs)
    assert np.allclose(result, outputs)


def test_Slice():
    x = np.random.randn(20, 10, 5).astype(np.float32)
    starts = np.array([0, 0], dtype=np.int64)
    ends = np.array([3, 10], dtype=np.int64)
    axes = np.array([0, 1], dtype=np.int64)
    steps = np.array([1, 1], dtype=np.int64)
    y = x[0:3, 0:10]

    op = Slice(x, starts, ends, axes=axes, steps=steps)
    check(op, [], [y])

    # non-constant input tensor
    x_input_op = Input((20, 10, 5), np.dtype(np.float32))
    op = Slice(x_input_op, starts, ends, axes=axes, steps=steps)
    check(op, [x], [y])

    # non-constant starts input
    starts_input_op = Input((2,), np.dtype(np.int64))
    op = Slice(x, starts_input_op, ends, axes=axes, steps=steps)
    check(op, [starts], [y])

    # non-constant ends input
    ends_input_op = Input((2,), np.dtype(np.int64))
    op = Slice(x, starts, ends_input_op, axes=axes, steps=steps)
    check(op, [ends], [y])

    # non-constant axes input
    axes_input_op = Input((2,), np.dtype(np.int64))
    op = Slice(x, starts, ends, axes=axes_input_op, steps=steps)
    check(op, [axes], [y])

    # non-constant steps input
    steps_input_op = Input((2,), np.dtype(np.int64))
    op = Slice(x, starts, ends, axes=axes, steps=steps_input_op)
    check(op, [steps], [y])


def test_Slice_default_axes():
    x = np.random.randn(20, 10, 5).astype(np.float32)
    starts = np.array([0, 0, 3], dtype=np.int64)
    ends = np.array([20, 10, 4], dtype=np.int64)
    y = x[:, :, 3:4]

    op = Slice(x, starts, ends)
    check(op, [], [y])


def test_Slice_default_steps():
    x = np.random.randn(20, 10, 5).astype(np.float32)
    starts = np.array([0, 0, 3], dtype=np.int64)
    ends = np.array([20, 10, 4], dtype=np.int64)
    axes = np.array([0, 1, 2], dtype=np.int64)
    y = x[:, :, 3:4]

    op = Slice(x, starts, ends, axes=axes)
    check(op, [], [y])


def test_Slice_end_out_of_bounds():
    x = np.random.randn(20, 10, 5).astype(np.float32)
    starts = np.array([1], dtype=np.int64)
    ends = np.array([1000], dtype=np.int64)
    axes = np.array([1], dtype=np.int64)
    steps = np.array([1], dtype=np.int64)
    y = x[:, 1:1000]

    op = Slice(x, starts, ends, axes=axes, steps=steps)
    check(op, [], [y])


def test_Slice_neg():
    x = np.random.randn(20, 10, 5).astype(np.float32)
    starts = np.array([0], dtype=np.int64)
    ends = np.array([-1], dtype=np.int64)
    axes = np.array([1], dtype=np.int64)
    steps = np.array([1], dtype=np.int64)
    y = x[:, 0:-1]

    op = Slice(x, starts, ends, axes=axes, steps=steps)
    check(op, [], [y])


def test_Slice_neg_steps():
    x = np.random.randn(20, 10, 5).astype(np.float32)
    starts = np.array([20, 10, 4], dtype=np.int64)
    ends = np.array([0, 0, 1], dtype=np.int64)
    axes = np.array([0, 1, 2], dtype=np.int64)
    steps = np.array([-1, -3, -2]).astype(np.int64)
    y = x[20:0:-1, 10:0:-3, 4:1:-2]

    op = Slice(x, starts, ends, axes=axes, steps=steps)
    check(op, [], [y])


def test_Slice_negative_axes():
    x = np.random.randn(20, 10, 5).astype(np.float32)
    starts = np.array([0, 0, 3], dtype=np.int64)
    ends = np.array([20, 10, 4], dtype=np.int64)
    axes = np.array([0, -2, -1], dtype=np.int64)
    y = x[:, :, 3:4]

    op = Slice(x, starts, ends, axes=axes)
    check(op, [], [y])


def test_Slice_start_out_of_bounds():
    x = np.random.randn(20, 10, 5).astype(np.float32)
    starts = np.array([1000], dtype=np.int64)
    ends = np.array([1000], dtype=np.int64)
    axes = np.array([1], dtype=np.int64)
    steps = np.array([1], dtype=np.int64)
    y = x[:, 1000:1000]

    op = Slice(x, starts, ends, axes=axes, steps=steps)
    check(op, [], [y])


def test_Slice_unordered_axes():
    x = np.random.randn(20, 10, 5).astype(np.float32)
    starts = np.array([0, 3, 0], dtype=np.int64)
    ends = np.array([10, 4, 20], dtype=np.int64)
    axes = np.array([-2, 2, 0], dtype=np.int64)
    y = x[:, :, 3:4]

    op = Slice(x, starts, ends, axes=axes)
    check(op, [], [y])
