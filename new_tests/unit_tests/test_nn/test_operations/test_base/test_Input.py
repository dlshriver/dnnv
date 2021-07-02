import numpy as np
import onnx
import onnx.helper

from dnnv.nn.operations.base import *


def test_Input():
    input_op = Input((-1, 5), np.dtype(np.float32))


def test_from_onnx_0():
    onnx_node = onnx.helper.make_tensor_value_info(
        "test_input", onnx.TensorProto.FLOAT, ["N", 3, 4, 4]
    )
    input_op_from_onnx = Input.from_onnx(onnx_node)
    input_op_from_init = Input(np.array([-1, 3, 4, 4]), np.dtype(np.float32))
    assert type(input_op_from_onnx) == type(input_op_from_init)
    assert np.all(input_op_from_onnx.shape == input_op_from_init.shape)
    assert input_op_from_onnx.dtype == input_op_from_init.dtype


def test_from_onnx_1():
    onnx_node = onnx.helper.make_tensor_value_info(
        "test_input", onnx.TensorProto.INT32, [1, 3, 4, 4]
    )
    input_op_from_onnx = Input.from_onnx(onnx_node)
    input_op_from_init = Input(np.array([1, 3, 4, 4]), np.dtype(np.int32))
    assert type(input_op_from_onnx) == type(input_op_from_init)
    assert np.all(input_op_from_onnx.shape == input_op_from_init.shape)
    assert input_op_from_onnx.dtype == input_op_from_init.dtype
