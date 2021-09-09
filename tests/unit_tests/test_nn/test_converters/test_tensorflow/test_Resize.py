import numpy as np
import pytest

from dnnv.nn.converters.tensorflow import *
from dnnv.nn.operations import *


def test_Resize():
    data = np.array(
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
    scales = np.array([1.0, 1.0, 0.8, 0.8], dtype=np.float32)

    op = Resize(data, np.array([]), scales, np.array([]))
    tf_op = TensorflowConverter().visit(op)
    with pytest.raises(TensorflowConverterError):
        result = tf_op().numpy()

    op = Resize(
        Input((1, 1, 4, 4), np.dtype(np.float32)),
        np.array([]),
        scales,
        np.array([]),
    )
    tf_op = TensorflowConverter().visit(op)
    with pytest.raises(TensorflowConverterError):
        result = tf_op(data).numpy()


def test_Resize_tf_crop_and_resize():
    data = np.array(
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
    roi = np.array([0, 0, 0.4, 0.6, 1, 1, 0.6, 0.8], dtype=np.float32)
    sizes = np.array([1, 1, 3, 3], dtype=np.int64)
    y = np.array([[[[7.6000004, 7.9, 8.2], [8.8, 9.1, 9.400001], [10.0, 10.3, 10.6]]]])

    op = Resize(
        data,
        roi,
        np.array([]),
        sizes,
        coordinate_transformation_mode="tf_crop_and_resize",
        mode="linear",
    )
    tf_op = TensorflowConverter().visit(op)
    result = tf_op().numpy()
    assert np.allclose(result, y)


def test_Resize_tf_crop_and_resize_extrapolation_value():
    data = np.array(
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
    roi = np.array([0, 0, 0.4, 0.6, 1, 1, 1.2, 1.7], dtype=np.float32)
    sizes = np.array([1, 1, 3, 3], dtype=np.int64)
    y = np.array(
        [[[[7.6000004, 10.0, 10.0], [12.400001, 10.0, 10.0], [10.0, 10.0, 10.0]]]]
    )

    op = Resize(
        data,
        roi,
        np.array([]),
        sizes,
        coordinate_transformation_mode="tf_crop_and_resize",
        mode="linear",
        extrapolation_value=10.0,
    )
    tf_op = TensorflowConverter().visit(op)
    result = tf_op().numpy()
    assert np.allclose(result, y)
