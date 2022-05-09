import numpy as np
import onnxruntime
import pytest

from dnnv.nn.converters.onnx import *
from dnnv.nn.operations import *

# Tests based on:
# https://github.com/onnx/onnx/blob/2ab133404afce34552aaccd86e7023e1fb9a60d2/onnx/test/shape_inference_test.py
# https://github.com/onnx/onnx/blob/2ab133404afce34552aaccd86e7023e1fb9a60d2/onnx/test/automatic_upgrade_test.py
# https://github.com/onnx/onnx/blob/35092895d9bf3592e58f4710d098f8131afef259/onnx/backend/test/case/node/split.py


def test_Split_export_1d() -> None:
    input = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).astype(np.float32)

    op = Split(input, axis=0, split=[2, 2, 2])
    all_results = np.zeros((3, 2))
    for i in range(3):
        outputselect = OutputSelect(op, i)
        onnx_model = convert(OperationGraph([outputselect]))
        results = onnxruntime.backend.run(onnx_model, [])
        all_results[i] = results

    expected_outputs = [
        np.array([1.0, 2.0]).astype(np.float32),
        np.array([3.0, 4.0]).astype(np.float32),
        np.array([5.0, 6.0]).astype(np.float32),
    ]
    assert len(all_results) == 3
    assert np.allclose(all_results, expected_outputs)

    op = Split(input, axis=0, split=np.array([2, 4]).astype(np.int64))
    onnx_model = convert(OperationGraph([op]))
    results = onnxruntime.backend.run(onnx_model, [])

    expected_outputs = [
        np.array([1.0, 2.0]).astype(np.float32),
        np.array([3.0, 4.0, 5.0, 6.0]).astype(np.float32),
    ]
    assert len(results) == 2
    assert np.allclose(results, expected_outputs)


def test_Split_export_2d() -> None:
    input = np.array(
        [[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [7.0, 8.0, 9.0, 10.0, 11.0, 12.0]]
    ).astype(np.float32)

    # node = onnx.helper.make_node(
    #     'Split',
    #     inputs=['input'],
    #     outputs=['output_1', 'output_2'],
    #     axis=1
    # )
    op = Split(input, axis=1, split=[2, 2])
    all_results = []
    for i in range(2):
        outputselect = OutputSelect(op, i)
        onnx_model = convert(OperationGraph([outputselect]))
        results = onnxruntime.backend.run(onnx_model, [])
        all_results.append(results)
    expected_outputs = [
        np.array([[1.0, 2.0, 3.0], [7.0, 8.0, 9.0]]).astype(np.float32),
        np.array([[4.0, 5.0, 6.0], [10.0, 11.0, 12.0]]).astype(np.float32),
    ]

    # expect(node, inputs=[input], outputs=[y for y in expected_outputs], name='test_split_equal_parts_2d')
    for i in range(2):
        assert all_results[i].shape == (2, 3)
        assert np.allclose(all_results[i], expected_outputs[i])
    split = np.array([2, 4]).astype(np.int64)
    # node = onnx.helper.make_node(
    #     'Split',
    #     inputs=['input', 'split'],
    #     outputs=['output_1', 'output_2'],
    #     axis=1,
    # )

    expected_outputs = [
        np.array([[1.0, 2.0], [7.0, 8.0]]).astype(np.float32),
        np.array([[3.0, 4.0, 5.0, 6.0], [9.0, 10.0, 11.0, 12.0]]).astype(np.float32),
    ]

    # expect(node, inputs=[input, split], outputs=[y for y in expected_outputs], name='test_split_variable_parts_2d')


def test_Split_export_default_values() -> None:
    input = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).astype(np.float32)

    # If axis is not specified, split is applied on default axis 0
    node = onnx.helper.make_node(
        "Split", inputs=["input"], outputs=["output_1", "output_2", "output_3"]
    )

    expected_outputs = [
        np.array([1.0, 2.0]).astype(np.float32),
        np.array([3.0, 4.0]).astype(np.float32),
        np.array([5.0, 6.0]).astype(np.float32),
    ]
    expect(
        node,
        inputs=[input],
        outputs=[y for y in expected_outputs],
        name="test_split_equal_parts_default_axis",
    )

    split = np.array([2, 4]).astype(np.int64)
    node = onnx.helper.make_node(
        "Split", inputs=["input", "split"], outputs=["output_1", "output_2"]
    )

    expected_outputs = [
        np.array([1.0, 2.0]).astype(np.float32),
        np.array([3.0, 4.0, 5.0, 6.0]).astype(np.float32),
    ]
    expect(
        node,
        inputs=[input, split],
        outputs=[y for y in expected_outputs],
        name="test_split_variable_parts_default_axis",
    )


def test_Split_export_zero_size_splits() -> None:
    input = np.array([]).astype(np.float32)

    # Split emtpy tensor to tensors of size zero
    split = np.array([0, 0, 0]).astype(np.int64)
    node = onnx.helper.make_node(
        "Split", inputs=["input", "split"], outputs=["output_1", "output_2", "output_3"]
    )

    expected_outputs = [
        np.array([]).astype(np.float32),
        np.array([]).astype(np.float32),
        np.array([]).astype(np.float32),
    ]
    expect(
        node,
        inputs=[input, split],
        outputs=[y for y in expected_outputs],
        name="test_split_zero_size_splits",
    )
