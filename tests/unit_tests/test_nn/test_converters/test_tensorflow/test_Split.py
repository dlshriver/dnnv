import numpy as np
import pytest

from dnnv.nn.converters.tensorflow import *
from dnnv.nn.operations import *


def test_Split_export_1d() -> None:
    input = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).astype(np.float32)
    op = Split(input, axis=0, split=np.array([2, 2, 2]))
    tf_op = TensorflowConverter().visit(op)
    result_ = tf_op()
    expected_outputs = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]).astype(np.float32)
    assert len(result_) == 3
    assert np.array_equiv(result_, expected_outputs)

    op = Split(input, axis=0, split=np.array([2, 4]).astype(np.int64))
    tf_op = TensorflowConverter().visit(op)
    result_ = tf_op()
    assert len(result_) == 2
    assert np.array_equiv(result_[0], np.array([1.0, 2.0]).astype(np.float32))
    assert np.array_equiv(result_[1], np.array([3.0, 4.0, 5.0, 6.0]).astype(np.float32))


def test_Split_export_2d() -> None:
    input = np.array(
        [[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [7.0, 8.0, 9.0, 10.0, 11.0, 12.0]]
    ).astype(np.float32)

    op = Split(input, axis=1, split=np.array([3, 3]))
    tf_op = TensorflowConverter().visit(op)
    result_ = tf_op()
    expected_outputs = np.array(
        [[[1.0, 2.0, 3.0], [7.0, 8.0, 9.0]], [[4.0, 5.0, 6.0], [10.0, 11.0, 12.0]]]
    ).astype(np.float32)
    for i in range(2):
        assert result_[i].shape == (2, 3)
        assert np.array_equiv(result_[i], expected_outputs[i])

    op = Split(input, axis=1, split=np.array([2, 4]))
    tf_op = TensorflowConverter().visit(op)
    result_ = tf_op()
    expected_outputs1 = np.array([[1.0, 2.0], [7.0, 8.0]]).astype(np.float32)
    expected_outputs2 = np.array(
        [[3.0, 4.0, 5.0, 6.0], [9.0, 10.0, 11.0, 12.0]]
    ).astype(np.float32)
    assert result_[0].shape == (2, 2)
    assert np.array_equiv(result_[0], expected_outputs1)
    assert result_[1].shape == (2, 4)
    assert np.array_equiv(result_[1], expected_outputs2)


def test_Split_export_default_values() -> None:
    input = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).astype(np.float32)
    op = Split(input, split=np.array([2, 2, 2]))
    tf_op = TensorflowConverter().visit(op)
    result_ = tf_op()
    expected_outputs = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]).astype(np.float32)
    assert len(result_) == 3
    assert np.array(result_).shape == expected_outputs.shape
    assert np.array_equiv(np.array(result_), expected_outputs)

    op = Split(input, split=np.array([2, 4]))
    tf_op = TensorflowConverter().visit(op)
    result_ = tf_op()
    assert len(result_) == 2
    assert len(result_[0]) == 2
    assert len(result_[1]) == 4
    assert np.array_equiv(result_[0], np.array([1.0, 2.0]).astype(np.float32))
    assert np.array_equiv(result_[1], np.array([3.0, 4.0, 5.0, 6.0]).astype(np.float32))


def test_Split_export_zero_size_splits() -> None:
    input = np.array([]).astype(np.float32)
    op = Split(input, split=np.array([0, 0, 0]))
    tf_op = TensorflowConverter().visit(op)
    result_ = tf_op()
    expected_outputs = np.array([[], [], []]).astype(np.float32)
    assert np.array(result_).shape == (3, 0)
    assert np.array_equiv(np.array(result_), expected_outputs)
