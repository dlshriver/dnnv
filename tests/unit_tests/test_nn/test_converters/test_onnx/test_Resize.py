import numpy as np
import onnxruntime.backend

from dnnv.nn.converters.onnx import OnnxConverter
from dnnv.nn.graph import OperationGraph
from dnnv.nn.operations import *


def test_Resize_downsample_scales_nearest():
    data = np.array(
        [
            [
                [
                    [1, 2, 3, 4],
                    [5, 6, 7, 8],
                ]
            ]
        ],
        dtype=np.float32,
    )
    scales = np.array([1.0, 1.0, 0.6, 0.6], dtype=np.float32)
    y = np.array([[[[1, 3]]]], dtype=np.float32)

    op = Resize(data, np.array([]), scales, np.array([]))
    onnx_model = OnnxConverter.convert(OperationGraph([op]))
    result = onnxruntime.backend.run(onnx_model, [])[0]
    assert np.allclose(result, y)

    op = Resize(
        Input((1, 1, 2, 4), np.dtype(np.float32)),
        np.array([]),
        scales,
        np.array([]),
    )
    onnx_model = OnnxConverter.convert(OperationGraph([op]))
    result = onnxruntime.backend.run(onnx_model, [data])[0]
    assert np.allclose(result, y)


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
    y = np.array([[[[7.6, 7.9, 8.2], [8.8, 9.1, 9.4], [10.0, 10.3, 10.6]]]])

    op = Resize(
        data,
        roi,
        np.array([]),
        sizes,
        coordinate_transformation_mode="tf_crop_and_resize",
        mode="linear",
    )
    onnx_model = OnnxConverter.convert(OperationGraph([op]))
    result = onnxruntime.backend.run(onnx_model, [])[0]
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
    y = np.array([[[[7.6, 10.0, 10.0], [12.4, 10.0, 10.0], [10.0, 10.0, 10.0]]]])

    op = Resize(
        data,
        roi,
        np.array([]),
        sizes,
        coordinate_transformation_mode="tf_crop_and_resize",
        mode="linear",
        extrapolation_value=10.0,
    )
    onnx_model = OnnxConverter.convert(OperationGraph([op]))
    result = onnxruntime.backend.run(onnx_model, [])[0]
    assert np.allclose(result, y)
