import numpy as np
import pytest

from dnnv.nn.converters.tensorflow import *
from dnnv.nn.operations import *

TOL = 1e-6


def test_AveragePool_1d_default_const():
    op = AveragePool(
        np.array([[[1, 2, 3], [4, 5, 6], [7, 8, 9]]], dtype=np.float32), np.array([2])
    )
    tf_op = TensorflowConverter().visit(op)
    result = tf_op().numpy()
    y = np.array([[[1.5, 2.5], [4.5, 5.5], [7.5, 8.5]]], dtype=np.float32)
    assert np.all(result >= (y - TOL))
    assert np.all(result <= (y + TOL))


def test_AveragePool_1d_default():
    input_op = Input((1, 3, 3), np.dtype(np.float32))
    op = AveragePool(input_op, np.array([2]))
    tf_op = TensorflowConverter().visit(op)
    x = np.array([[[1, 2, 3], [4, 5, 6], [7, 8, 9]]], dtype=np.float32)
    result = tf_op(x).numpy()
    y = np.array([[[1.5, 2.5], [4.5, 5.5], [7.5, 8.5]]], dtype=np.float32)
    assert np.all(result >= (y - TOL))
    assert np.all(result <= (y + TOL))


def test_AveragePool_2d_ceil():
    input_op = Input((1, 1, 4, 4), np.dtype(np.float32))
    op = AveragePool(input_op, np.array([3, 3]), strides=[2, 2], ceil_mode=True)
    tf_op = TensorflowConverter().visit(op)
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
        ],
        dtype=np.float32,
    )
    with pytest.raises(TensorflowConverterError):
        result = tf_op(x).numpy()
    # y = np.array([[[[6, 7.5], [12, 13.5]]]], dtype=np.float32)
    # print(result)
    # assert np.all(result >= (y - TOL))
    # assert np.all(result <= (y + TOL))


def test_AveragePool_2d_default():
    input_op = Input((1, 3, 2, 2), np.dtype(np.float32))
    op = AveragePool(input_op, np.array([2, 2]))
    tf_op = TensorflowConverter().visit(op)
    x = np.array(
        [[[[1, -6], [2, -5]], [[3, -4], [4, -3]], [[5, -2], [6, -1]]]], dtype=np.float32
    )
    result = tf_op(x).numpy()
    y = np.array([[[[-2]], [[0]], [[2]]]], dtype=np.float32)
    assert np.all(result >= (y - TOL))
    assert np.all(result <= (y + TOL))


def test_AveragePool_2d_pads():
    input_op = Input((1, 3, 28, 28), np.dtype(np.float32))
    op = AveragePool(input_op, np.array([3, 3]), pads=np.array([2, 2, 2, 2]))
    tf_op = TensorflowConverter().visit(op)
    x = np.random.randn(1, 3, 28, 28).astype(np.float32)
    with pytest.raises(TensorflowConverterError):
        result = tf_op(x).numpy()
    # expected_out_shape = (1, 3, 30, 30)
    # assert result.shape == expected_out_shape


def test_AveragePool_2d_pads_count_include_pad():
    input_op = Input((1, 3, 28, 28), np.dtype(np.float32))
    op = AveragePool(
        input_op, np.array([3, 3]), pads=np.array([2, 2, 2, 2]), count_include_pad=True
    )
    tf_op = TensorflowConverter().visit(op)
    x = np.random.randn(1, 3, 28, 28).astype(np.float32)
    result = tf_op(x).numpy()
    expected_out_shape = (1, 3, 30, 30)
    assert result.shape == expected_out_shape


def test_AveragePool_2d_precomputed_pads():
    input_op = Input((1, 1, 5, 5), np.dtype(np.float32))
    op = AveragePool(input_op, np.array([5, 5]), pads=[2, 2, 2, 2])
    tf_op = TensorflowConverter().visit(op)
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
        ],
        dtype=np.float32,
    )
    with pytest.raises(TensorflowConverterError):
        result = tf_op(x).numpy()
    # y = np.array(
    #     [
    #         [
    #             [
    #                 [7, 7.5, 8, 8.5, 9],
    #                 [9.5, 10, 10.5, 11, 11.5],
    #                 [12, 12.5, 13, 13.5, 14],
    #                 [14.5, 15, 15.5, 16, 16.5],
    #                 [17, 17.5, 18, 18.5, 19],
    #             ]
    #         ]
    #     ],
    #     dtype=np.float32,
    # )
    # assert np.all(result >= (y - TOL))
    # assert np.all(result <= (y + TOL))


def test_AveragePool_2d_precomputed_pads_include_pad():
    input_op = Input((1, 1, 5, 5), np.dtype(np.float32))
    op = AveragePool(
        input_op, np.array([5, 5]), pads=[2, 2, 2, 2], count_include_pad=True
    )
    tf_op = TensorflowConverter().visit(op)
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
        ],
        dtype=np.float32,
    )
    result = tf_op(x).numpy()
    y = np.array(
        [
            [
                [
                    [2.5200, 3.6000, 4.8000, 4.0800, 3.2400],
                    [4.5600, 6.4000, 8.4000, 7.0400, 5.5200],
                    [7.2000, 10.0000, 13.0000, 10.8000, 8.4000],
                    [6.9600, 9.6000, 12.4000, 10.2400, 7.9200],
                    [6.1200, 8.4000, 10.8000, 8.8800, 6.8400],
                ]
            ]
        ],
        dtype=np.float32,
    )
    assert np.all(result >= (y - TOL))
    assert np.all(result <= (y + TOL))


def test_AveragePool_2d_precomputed_strides():
    input_op = Input((1, 1, 5, 5), np.dtype(np.float32))
    op = AveragePool(input_op, np.array([2, 2]), strides=[2, 2])
    tf_op = TensorflowConverter().visit(op)
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
        ],
        dtype=np.float32,
    )
    result = tf_op(x).numpy()
    y = np.array(
        [[[[4, 6], [14, 16]]]],
        dtype=np.float32,
    )
    assert np.all(result >= (y - TOL))
    assert np.all(result <= (y + TOL))


def test_AveragePool_2d_strides():
    input_op = Input((1, 3, 32, 32), np.dtype(np.float32))
    op = AveragePool(input_op, np.array([5, 5]), strides=[3, 3])
    tf_op = TensorflowConverter().visit(op)
    x = np.random.randn(1, 3, 32, 32).astype(np.float32)
    result = tf_op(x).numpy()
    expected_output_shape = (1, 3, 10, 10)
    assert result.shape == expected_output_shape


def test_AveragePool_3d_default():
    input_op = Input((1, 3, 32, 32, 32), np.dtype(np.float32))
    op = AveragePool(input_op, np.array([2, 2, 2]))
    tf_op = TensorflowConverter().visit(op)
    x = np.random.randn(1, 3, 32, 32, 32).astype(np.float32)
    result = tf_op(x).numpy()
    expected_output_shape = (1, 3, 31, 31, 31)
    assert result.shape == expected_output_shape
