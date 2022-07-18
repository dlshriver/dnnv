import numpy as np
import onnxruntime.backend

from dnnv.nn.converters.onnx import OnnxConverter
from dnnv.nn.graph import OperationGraph
from dnnv.nn.operations import *


def test_Split_export_1d() -> None:
    input = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).astype(np.float32)
    op = Split(input, axis=0, split=np.array([2, 2, 2]))
    all_results_list = []
    for i in range(3):
        onnx_model = OnnxConverter.convert(OperationGraph([OutputSelect(op, i)]))
        results = onnxruntime.backend.run(onnx_model, [])
        all_results_list.append(results[0])
    all_results = np.array(all_results_list)
    expected_outputs = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]).astype(np.float32)
    assert len(all_results) == 3
    assert np.array_equiv(all_results, expected_outputs)

    op = Split(input, axis=0, split=np.array([2, 4]).astype(np.int64))
    all_results_list = []
    for i in range(2):
        onnx_model = OnnxConverter.convert(OperationGraph([OutputSelect(op, i)]))
        results = onnxruntime.backend.run(onnx_model, [])
        all_results_list.append(results[0])
    all_results = np.array(all_results_list)
    assert len(all_results) == 2
    assert np.array_equiv(all_results[0], np.array([1.0, 2.0]).astype(np.float32))
    assert np.array_equiv(
        all_results[1], np.array([3.0, 4.0, 5.0, 6.0]).astype(np.float32)
    )


def test_Split_export_2d() -> None:
    input = np.array(
        [[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [7.0, 8.0, 9.0, 10.0, 11.0, 12.0]]
    ).astype(np.float32)

    op = Split(input, axis=1, split=np.array([3, 3]))
    all_results_list = []
    for i in range(2):
        outputselect = OutputSelect(op, i)
        onnx_model = OnnxConverter.convert(OperationGraph([outputselect]))
        results = onnxruntime.backend.run(onnx_model, [])
        all_results_list.append(results[0])
    expected_outputs = np.array(
        [[[1.0, 2.0, 3.0], [7.0, 8.0, 9.0]], [[4.0, 5.0, 6.0], [10.0, 11.0, 12.0]]]
    ).astype(np.float32)
    for i in range(2):
        assert all_results_list[i].shape == (2, 3)
        assert np.array_equiv(all_results_list[i], expected_outputs[i])

    op = Split(input, axis=1, split=np.array([2, 4]))
    all_results_list = []
    for i in range(2):
        outputselect = OutputSelect(op, i)
        onnx_model = OnnxConverter.convert(OperationGraph([outputselect]))
        results = onnxruntime.backend.run(onnx_model, [])
        all_results_list.append(results[0])
    expected_outputs1 = np.array([[1.0, 2.0], [7.0, 8.0]]).astype(np.float32)
    expected_outputs2 = np.array(
        [[3.0, 4.0, 5.0, 6.0], [9.0, 10.0, 11.0, 12.0]]
    ).astype(np.float32)
    assert all_results_list[0].shape == (2, 2)
    assert np.array_equiv(all_results_list[0], expected_outputs1)
    assert all_results_list[1].shape == (2, 4)
    assert np.array_equiv(all_results_list[1], expected_outputs2)


def test_Split_export_default_values() -> None:
    input = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).astype(np.float32)
    op = Split(input, split=np.array([2, 2, 2]))
    all_results_list = []
    for i in range(3):
        onnx_model = OnnxConverter.convert(OperationGraph([OutputSelect(op, i)]))
        results = onnxruntime.backend.run(onnx_model, [])
        all_results_list.append(results[0])
    all_results = np.array(all_results_list)
    expected_outputs = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]).astype(np.float32)
    assert len(all_results) == 3
    assert all_results.shape == expected_outputs.shape
    assert np.array_equiv(all_results, expected_outputs)

    op = Split(input, split=np.array([2, 4]))
    all_results_list = []
    for i in range(2):
        onnx_model = OnnxConverter.convert(OperationGraph([OutputSelect(op, i)]))
        results = onnxruntime.backend.run(onnx_model, [])
        all_results_list.append(results[0])
    assert len(all_results_list) == 2
    assert len(all_results_list[0]) == 2
    assert len(all_results_list[1]) == 4
    assert np.array_equiv(all_results_list[0], np.array([1.0, 2.0]).astype(np.float32))
    assert np.array_equiv(
        all_results_list[1], np.array([3.0, 4.0, 5.0, 6.0]).astype(np.float32)
    )


def test_Split_export_zero_size_splits() -> None:
    # Split emtpy tensor to tensors of size zero
    input = np.array([]).astype(np.float32)
    op = Split(input, split=np.array([0, 0, 0]))
    all_results_list = []
    for i in range(3):
        onnx_model = OnnxConverter.convert(OperationGraph([OutputSelect(op, i)]))
        results = onnxruntime.backend.run(onnx_model, [])
        all_results_list.append(results[0])
    all_results = np.array(all_results_list)
    expected_outputs = np.array([[], [], []]).astype(np.float32)
    assert all_results.shape == (3, 0)
    assert np.array_equiv(all_results, expected_outputs)
