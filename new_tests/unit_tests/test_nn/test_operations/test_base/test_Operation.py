import numpy as np
import onnx
import onnx.helper
import pytest

from dnnv.nn.operations import *
from dnnv.nn.operations.base import *


def test_str():
    assert str(Operation()) == "Operation"
    assert str(Input(None, None)) == "Input"
    assert str(Add(None, None)) == "Add"
    assert str(Cast(None, None)) == "Cast"
    assert str(Gemm(np.empty((1, 5)), np.empty((5, 2)))) == "Gemm"


def test_from_onnx_Input():
    onnx_node = onnx.helper.make_tensor_value_info(
        "test_input", onnx.TensorProto.FLOAT, ["N", 3, 4, 4]
    )
    op_from_onnx = Operation.from_onnx(onnx_node)
    input_op_from_init = Input(np.array([-1, 3, 4, 4]), np.dtype(np.float32))
    assert type(op_from_onnx) is Input
    assert np.all(op_from_onnx.shape == input_op_from_init.shape)
    assert op_from_onnx.dtype == input_op_from_init.dtype


def test_from_onnx_Add():
    input_op = Input(np.array([-1, 5]), np.dtype(np.float32))

    add_node = onnx.helper.make_node(
        "Add", inputs=["input", "const"], outputs=["add"], name="add"
    )
    op_from_onnx = Operation.from_onnx(
        add_node, input_op, np.ones((1, 5), dtype=np.float32)
    )

    assert type(op_from_onnx) is Add
    assert op_from_onnx.a is input_op
    assert np.all(op_from_onnx.b == np.ones((1, 5)))


def test_from_onnx_Mul_constants(caplog):
    # TODO : is this a necessary feature?
    add_node = onnx.helper.make_node(
        "Mul", inputs=["const1", "const2"], outputs=["mul"], name="mul"
    )
    op_from_onnx = Operation.from_onnx(add_node, 2, 5)

    assert type(op_from_onnx) is Mul
    assert op_from_onnx.a == 2
    assert op_from_onnx.b == 5

    assert "Operation on constant inputs returned non-constant." in caplog.text


def test_from_onnx_Elu():
    input_op = Input(np.array([-1, 5]), np.dtype(np.float32))

    add_node = onnx.helper.make_node(
        "Elu", inputs=["input"], outputs=["elu"], name="elu", alpha=0.5
    )
    op_from_onnx = Operation.from_onnx(add_node, input_op)

    assert type(op_from_onnx) is Elu
    assert op_from_onnx.x is input_op
    assert op_from_onnx.alpha == 0.5


def test_from_onnx_Elu():
    input_op = Input(np.array([-1, 5]), np.dtype(np.float32))

    add_node = onnx.helper.make_node(
        "Fake", inputs=["input"], outputs=["fake"], name="fake"
    )
    with pytest.raises(ValueError) as excinfo:
        op_from_onnx = Operation.from_onnx(add_node, input_op)
    assert str(excinfo.value) == "Unimplemented operation type: Fake"


def test_inputs():
    input_op_0 = Input((1, 4), np.dtype(np.float32))
    add_op = Add(input_op_0, np.ones((1, 4), dtype=np.float32))
    add_op_inputs = add_op.inputs
    assert len(add_op_inputs) == 1
    assert input_op_0 in add_op_inputs

    input_op_1 = Input((1, 3, 2, 2), np.dtype(np.float32))
    flatten_op = Flatten(input_op_1)
    concat_op = Concat([add_op, flatten_op], axis=0)
    concat_op_inputs = concat_op.inputs
    assert len(concat_op_inputs) == 2
    assert add_op in concat_op_inputs
    assert flatten_op in concat_op_inputs

    reshape_op = Reshape(input_op_1, [1, 12])
    reshape_op_inputs = reshape_op.inputs
    assert len(reshape_op_inputs) == 1
    assert input_op_1 in reshape_op_inputs


def test_match_none():
    assert list(Operation.match([])) == []
    assert list(Input.match([])) == []
    assert list(Add.match([])) == []


def test_match_true():
    assert list(Operation.match([Operation()])) == [[]]
    assert list(Input.match([Input(None, None)])) == [[]]

    input_op_0 = Input(None, None)
    input_op_1 = Input(None, None)
    add_op = Add(input_op_0, input_op_1)
    assert list(Add.match([add_op])) == [[input_op_0, input_op_1]]


def test_match_false():
    assert list(Input.match([Operation()])) == []
    assert list(Add.match([Input(None, None)])) == []

    input_op_0 = Input(None, None)
    input_op_1 = Input(None, None)
    add_op = Add(input_op_0, input_op_1)
    assert list(Add.match([add_op, input_op_0])) == []
