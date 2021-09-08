import pytest
import numpy as np

from dnnv.nn.operations import *
from dnnv.nn.visitors import PrintVisitor
from dnnv.utils import get_subclasses


def test_missing():
    class FakeOperation:
        pass

    with pytest.raises(ValueError) as excinfo:
        PrintVisitor().visit(FakeOperation())
    assert str(excinfo.value).startswith(
        "Operation not currently supported by PrintVisitor: "
    )


def test_all():
    print_visitor = PrintVisitor()
    for operation in get_subclasses(Operation):
        assert hasattr(print_visitor, f"visit_{operation}")


def test_Add(capsys):
    input_op = Input((1, 5), np.dtype(np.float32))
    add_op_0 = Add(input_op, np.float32(13))
    PrintVisitor().visit(add_op_0)
    captured = capsys.readouterr()
    expected_output = """\
Input_0                         : Input((1, 5), dtype=float32)
Add_0                           : Add(Input_0, float32_0)
"""
    assert captured.out == expected_output

    add_op_1 = Add(input_op, np.ones((1, 5)))
    PrintVisitor().visit(add_op_1)
    captured = capsys.readouterr()
    expected_output = """\
Input_0                         : Input((1, 5), dtype=float32)
Add_0                           : Add(Input_0, ndarray(shape=(1, 5)))
"""
    assert captured.out == expected_output


def test_Atan(capsys):
    input_op = Input((1, 5), np.dtype(np.float32))
    atan_op = Atan(input_op)
    PrintVisitor().visit(atan_op)
    captured = capsys.readouterr()
    expected_output = """\
Input_0                         : Input((1, 5), dtype=float32)
Atan_0                          : Atan(Input_0)
"""
    assert captured.out == expected_output


def test_AveragePool(capsys):
    input_op = Input((1, 3, 4, 4), np.dtype(np.float32))
    avgpool_op = AveragePool(input_op, (2, 2))
    PrintVisitor().visit(avgpool_op)
    captured = capsys.readouterr()
    expected_output = """\
Input_0                         : Input((1, 3, 4, 4), dtype=float32)
AveragePool_0                   : AveragePool(Input_0, (2, 2))
"""
    assert captured.out == expected_output


def test_BatchNormalization(capsys):
    input_op = Input((1, 3, 4, 4), np.dtype(np.float32))
    batchnorm_op = BatchNormalization(
        input_op, np.ones((3,)), np.zeros((3,)), np.ones((3,)), np.zeros((3,))
    )
    PrintVisitor().visit(batchnorm_op)
    captured = capsys.readouterr()
    expected_output = """\
Input_0                         : Input((1, 3, 4, 4), dtype=float32)
BatchNormalization_0            : BatchNormalization(Input_0)
"""
    assert captured.out == expected_output


def test_Cast(capsys):
    input_op = Input((1, 5), np.dtype(np.float32))
    cast_op = Cast(input_op, to=np.dtype(np.float64))
    PrintVisitor().visit(cast_op)
    captured = capsys.readouterr()
    expected_output = """\
Input_0                         : Input((1, 5), dtype=float32)
Cast_0                          : Cast(Input_0, to=float64)
"""
    assert captured.out == expected_output


def test_Concat(capsys):
    input_op = Input((1, 5), np.dtype(np.float32))
    concat_op = Concat([input_op, input_op], axis=1)
    PrintVisitor().visit(concat_op)
    captured = capsys.readouterr()
    expected_output = """\
Input_0                         : Input((1, 5), dtype=float32)
Concat_0                        : Concat(['Input_0', 'Input_0'])
"""
    assert captured.out == expected_output


def test_Conv(capsys):
    input_op = Input((1, 3, 4, 4), np.dtype(np.float32))
    conv_op = Conv(
        input_op,
        np.ones((3, 3, 2, 2), dtype=np.float32),
        np.zeros((2,), dtype=np.float32),
    )
    PrintVisitor().visit(conv_op)
    captured = capsys.readouterr()
    expected_output = """\
Input_0                         : Input((1, 3, 4, 4), dtype=float32)
Conv_0                          : Conv(Input_0, kernel_shape=[2, 2], strides=[1, 1], pads=[0, 0, 0, 0])
"""
    assert captured.out == expected_output


def test_ConvTranspose(capsys):
    input_op = Input((1, 3, 4, 4), np.dtype(np.float32))
    convt_op = ConvTranspose(
        input_op,
        np.ones((3, 3, 2, 2), dtype=np.float32),
        np.zeros((2,), dtype=np.float32),
    )
    PrintVisitor().visit(convt_op)
    captured = capsys.readouterr()
    expected_output = """\
Input_0                         : Input((1, 3, 4, 4), dtype=float32)
ConvTranspose_0                 : ConvTranspose(Input_0, kernel_shape=[2, 2], strides=[1, 1], pads=[0, 0, 0, 0], output_padding=[0, 0])
"""
    assert captured.out == expected_output


def test_Div(capsys):
    input_op = Input((1, 5), np.dtype(np.float32))
    div_op = Div(input_op, np.float32(2))
    PrintVisitor().visit(div_op)
    captured = capsys.readouterr()
    expected_output = """\
Input_0                         : Input((1, 5), dtype=float32)
Div_0                           : Div(Input_0, float32_0)
"""
    assert captured.out == expected_output


def test_Dropout(capsys):
    input_op = Input((1, 5), np.dtype(np.float32))
    dropout_op = Dropout(input_op, ratio=np.float32(0.5))
    PrintVisitor().visit(dropout_op)
    captured = capsys.readouterr()
    expected_output = """\
Input_0                         : Input((1, 5), dtype=float32)
Dropout_0                       : Dropout(Input_0, ratio=0.5)
"""
    assert captured.out == expected_output


def test_Elu(capsys):
    input_op = Input((1, 5), np.dtype(np.float32))
    elu_op_0 = Elu(input_op)
    PrintVisitor().visit(elu_op_0)
    captured = capsys.readouterr()
    expected_output = """\
Input_0                         : Input((1, 5), dtype=float32)
Elu_0                           : Elu(Input_0, alpha=1.0)
"""
    assert captured.out == expected_output

    elu_op_1 = Elu(input_op, alpha=0.2)
    PrintVisitor().visit(elu_op_1)
    captured = capsys.readouterr()
    expected_output = """\
Input_0                         : Input((1, 5), dtype=float32)
Elu_0                           : Elu(Input_0, alpha=0.2)
"""
    assert captured.out == expected_output


def test_Expand(capsys):
    input_op = Input((1, 5), np.dtype(np.float32))
    expand_op_0 = Expand(input_op, np.array([1, 10]))
    PrintVisitor().visit(expand_op_0)
    captured = capsys.readouterr()
    expected_output = """\
Input_0                         : Input((1, 5), dtype=float32)
Expand_0                        : Expand(Input_0, [ 1 10])
"""
    assert captured.out == expected_output

    input_op = Input((1, 3, 2, 2), np.dtype(np.float32))
    expand_op_1 = Expand(input_op, np.array([1, 6, 4, 4]))
    PrintVisitor().visit(expand_op_1)
    captured = capsys.readouterr()
    expected_output = """\
Input_0                         : Input((1, 3, 2, 2), dtype=float32)
Expand_0                        : Expand(Input_0, [1 6 4 4])
"""
    assert captured.out == expected_output


def test_Flatten(capsys):
    input_op = Input((1, 3, 2, 2), np.dtype(np.float32))
    flatten_op_0 = Flatten(input_op)
    PrintVisitor().visit(flatten_op_0)
    captured = capsys.readouterr()
    expected_output = """\
Input_0                         : Input((1, 3, 2, 2), dtype=float32)
Flatten_0                       : Flatten(Input_0, axis=1)
"""
    assert captured.out == expected_output

    flatten_op_1 = Flatten(input_op, axis=0)
    PrintVisitor().visit(flatten_op_1)
    captured = capsys.readouterr()
    expected_output = """\
Input_0                         : Input((1, 3, 2, 2), dtype=float32)
Flatten_0                       : Flatten(Input_0, axis=0)
"""
    assert captured.out == expected_output


def test_Gather(capsys):
    input_op = Input((1, 3, 2, 2), np.dtype(np.float32))
    gather_op_0 = Gather(input_op, np.array([[0], [0]]))
    PrintVisitor().visit(gather_op_0)
    captured = capsys.readouterr()
    expected_output = """\
Input_0                         : Input((1, 3, 2, 2), dtype=float32)
Gather_0                        : Gather(Input_0, [[0] [0]], axis=0)
"""
    assert captured.out == expected_output

    gather_op_0 = Gather(input_op, np.array([[0, 1, 2], [1, 2, 0], [2, 1, 0]]), axis=1)
    PrintVisitor().visit(gather_op_0)
    captured = capsys.readouterr()
    expected_output = """\
Input_0                         : Input((1, 3, 2, 2), dtype=float32)
Gather_0                        : Gather(Input_0, ndarray(shape=(3, 3)), axis=1)
"""
    assert captured.out == expected_output


def test_Gemm(capsys):
    input_op = Input((1, 5), np.dtype(np.float32))
    gemm_op_0 = Gemm(
        input_op, np.ones((5, 2), dtype=np.float32), np.zeros((2,), dtype=np.float32)
    )
    PrintVisitor().visit(gemm_op_0)
    captured = capsys.readouterr()
    expected_output = """\
Input_0                         : Input((1, 5), dtype=float32)
Gemm_0                          : Gemm(Input_0, ndarray(shape=(5, 2)), [0. 0.], transpose_a=0, transpose_b=0, alpha=1.000000, beta=1.000000)
"""
    assert captured.out == expected_output

    gemm_op_1 = Gemm(
        np.ones((2, 5), dtype=np.float32),
        input_op,
        np.zeros((2,), dtype=np.float32),
        transpose_b=True,
    )
    PrintVisitor().visit(gemm_op_1)
    captured = capsys.readouterr()
    expected_output = """\
Input_0                         : Input((1, 5), dtype=float32)
Gemm_0                          : Gemm(ndarray(shape=(2, 5)), Input_0, [0. 0.], transpose_a=0, transpose_b=1, alpha=1.000000, beta=1.000000)
"""
    assert captured.out == expected_output


def test_GlobalAveragePool(capsys):
    input_op = Input((1, 3, 4, 4), np.dtype(np.float32))
    globalavgpool_op = GlobalAveragePool(input_op)
    PrintVisitor().visit(globalavgpool_op)
    captured = capsys.readouterr()
    expected_output = """\
Input_0                         : Input((1, 3, 4, 4), dtype=float32)
GlobalAveragePool_0             : GlobalAveragePool(Input_0)
"""
    assert captured.out == expected_output


def test_Identity(capsys):
    input_op = Input((1, 5), np.dtype(np.float32))
    identity_op = Identity(input_op)
    PrintVisitor().visit(identity_op)
    captured = capsys.readouterr()
    expected_output = """\
Input_0                         : Input((1, 5), dtype=float32)
Identity_0                      : Identity(Input_0)
"""
    assert captured.out == expected_output


def test_Input(capsys):
    PrintVisitor().visit(Input((1, 2, 3, 4), np.dtype(np.float64)))
    captured = capsys.readouterr()
    expected_output = """\
Input_0                         : Input((1, 2, 3, 4), dtype=float64)
"""
    assert captured.out == expected_output


def test_LeakyRelu(capsys):
    input_op = Input((1, 5), np.dtype(np.float32))
    leakyrelu_op_0 = LeakyRelu(input_op)
    PrintVisitor().visit(leakyrelu_op_0)
    captured = capsys.readouterr()
    expected_output = """\
Input_0                         : Input((1, 5), dtype=float32)
LeakyRelu_0                     : LeakyRelu(Input_0, alpha=0.010000)
"""
    assert captured.out == expected_output

    leakyrelu_op_1 = LeakyRelu(input_op, alpha=0.2)
    PrintVisitor().visit(leakyrelu_op_1)
    captured = capsys.readouterr()
    expected_output = """\
Input_0                         : Input((1, 5), dtype=float32)
LeakyRelu_0                     : LeakyRelu(Input_0, alpha=0.200000)
"""
    assert captured.out == expected_output


def test_LogSoftmax(capsys):
    input_op = Input((1, 5), np.dtype(np.float32))
    logsoftmax_op_0 = LogSoftmax(input_op)
    PrintVisitor().visit(logsoftmax_op_0)
    captured = capsys.readouterr()
    expected_output = """\
Input_0                         : Input((1, 5), dtype=float32)
LogSoftmax_0                    : LogSoftmax(Input_0, axis=-1)
"""
    assert captured.out == expected_output

    logsoftmax_op_1 = LogSoftmax(input_op, axis=0)
    PrintVisitor().visit(logsoftmax_op_1)
    captured = capsys.readouterr()
    expected_output = """\
Input_0                         : Input((1, 5), dtype=float32)
LogSoftmax_0                    : LogSoftmax(Input_0, axis=0)
"""
    assert captured.out == expected_output


def test_MatMul(capsys):
    input_op = Input((1, 5), np.dtype(np.float32))
    matmul_op_0 = MatMul(input_op, np.ones((5, 2), dtype=np.float32))
    PrintVisitor().visit(matmul_op_0)
    captured = capsys.readouterr()
    expected_output = """\
Input_0                         : Input((1, 5), dtype=float32)
MatMul_0                        : MatMul(Input_0, ndarray(shape=(5, 2)))
"""
    assert captured.out == expected_output

    matmul_op_1 = MatMul(
        np.ones((2, 1), dtype=np.float32),
        input_op,
    )
    PrintVisitor().visit(matmul_op_1)
    captured = capsys.readouterr()
    expected_output = """\
Input_0                         : Input((1, 5), dtype=float32)
MatMul_0                        : MatMul([[1.] [1.]], Input_0)
"""
    assert captured.out == expected_output


def test_MaxPool(capsys):
    input_op = Input((1, 3, 4, 4), np.dtype(np.float32))
    maxpool_op = MaxPool(input_op, (2, 2))
    PrintVisitor().visit(maxpool_op)
    captured = capsys.readouterr()
    expected_output = """\
Input_0                         : Input((1, 3, 4, 4), dtype=float32)
MaxPool_0                       : MaxPool(Input_0)
"""
    assert captured.out == expected_output


def test_Mul(capsys):
    input_op = Input((1, 5), np.dtype(np.float32))
    mul_op = Mul(input_op, np.float32(2))
    PrintVisitor().visit(mul_op)
    captured = capsys.readouterr()
    expected_output = """\
Input_0                         : Input((1, 5), dtype=float32)
Mul_0                           : Mul(Input_0, float32_0)
"""
    assert captured.out == expected_output


def test_OutputSelect(capsys):
    input_op = Input((1, 5), np.dtype(np.float32))
    index_op = OutputSelect(input_op, (0, 1))
    PrintVisitor().visit(index_op)
    captured = capsys.readouterr()
    expected_output = """\
Input_0                         : Input((1, 5), dtype=float32)
OutputSelect_0                  : OutputSelect(Input_0, (0, 1))
"""
    assert captured.out == expected_output

    index_op = input_op[0, 4]
    PrintVisitor().visit(index_op)
    captured = capsys.readouterr()
    expected_output = """\
Input_0                         : Input((1, 5), dtype=float32)
OutputSelect_0                  : OutputSelect(Input_0, (0, 4))
"""
    assert captured.out == expected_output


def test_Pad(capsys):
    input_op = Input((1, 3, 4, 4), np.dtype(np.float32))
    pad_op = Pad(input_op, (0, 0, 0, 0))
    PrintVisitor().visit(pad_op)
    captured = capsys.readouterr()
    expected_output = """\
Input_0                         : Input((1, 3, 4, 4), dtype=float32)
Pad_0                           : Pad(Input_0, pads=(0, 0, 0, 0))
"""
    assert captured.out == expected_output


def test_Relu(capsys):
    input_op = Input((1, 5), np.dtype(np.float32))
    relu_op = Relu(input_op)
    PrintVisitor().visit(relu_op)
    captured = capsys.readouterr()
    expected_output = """\
Input_0                         : Input((1, 5), dtype=float32)
Relu_0                          : Relu(Input_0)
"""
    assert captured.out == expected_output


def test_Reshape(capsys):
    input_op = Input((1, 3, 2, 2), np.dtype(np.float32))
    reshape_op_0 = Reshape(input_op, np.array((1, 12)))
    PrintVisitor().visit(reshape_op_0)
    captured = capsys.readouterr()
    expected_output = """\
Input_0                         : Input((1, 3, 2, 2), dtype=float32)
Reshape_0                       : Reshape(Input_0, [ 1 12])
"""
    assert captured.out == expected_output

    reshape_op_1 = Reshape(input_op, np.array((1, 3, 2, 2)))
    PrintVisitor().visit(reshape_op_1)
    captured = capsys.readouterr()
    expected_output = """\
Input_0                         : Input((1, 3, 2, 2), dtype=float32)
Reshape_0                       : Reshape(Input_0, [1 3 2 2])
"""
    assert captured.out == expected_output


def test_Resize(capsys):
    input_op = Input((1, 3, 2, 2), np.dtype(np.float32))
    resize_op = Resize(
        input_op,
        np.array([], dtype=np.float32),
        np.array([1, 1, 2, 2], dtype=np.float64),
        np.array([], dtype=np.int64),
    )
    PrintVisitor().visit(resize_op)
    captured = capsys.readouterr()
    expected_output = """\
Input_0                         : Input((1, 3, 2, 2), dtype=float32)
Resize_0                        : Resize(Input_0, scales=[1.0, 1.0, 2.0, 2.0], coordinate_transformation_mode=half_pixel, cubic_coeff_a=-0.750000, exclude_outside=0, extrapolation_value=0.000000, mode=nearest, nearest_mode=round_prefer_floor)
"""
    assert captured.out == expected_output

    input_op = Input((1, 3, 2, 2), np.dtype(np.float32))
    resize_op = Resize(
        input_op,
        np.array([], dtype=np.float32),
        np.array([], dtype=np.float64),
        np.array([1, 3, 4, 4], dtype=np.int64),
    )
    PrintVisitor().visit(resize_op)
    captured = capsys.readouterr()
    expected_output = """\
Input_0                         : Input((1, 3, 2, 2), dtype=float32)
Resize_0                        : Resize(Input_0, sizes=[1, 3, 4, 4], coordinate_transformation_mode=half_pixel, cubic_coeff_a=-0.750000, exclude_outside=0, extrapolation_value=0.000000, mode=nearest, nearest_mode=round_prefer_floor)
"""
    assert captured.out == expected_output

    input_op = Input((1, 3, 2, 2), np.dtype(np.float32))
    resize_op = Resize(
        input_op,
        np.array([0, 0, 1, 1], dtype=np.float32),
        np.array([], dtype=np.float64),
        np.array([1, 3, 4, 4], dtype=np.int64),
    )
    PrintVisitor().visit(resize_op)
    captured = capsys.readouterr()
    expected_output = """\
Input_0                         : Input((1, 3, 2, 2), dtype=float32)
Resize_0                        : Resize(Input_0, roi=[0.0, 0.0, 1.0, 1.0], sizes=[1, 3, 4, 4], coordinate_transformation_mode=half_pixel, cubic_coeff_a=-0.750000, exclude_outside=0, extrapolation_value=0.000000, mode=nearest, nearest_mode=round_prefer_floor)
"""
    assert captured.out == expected_output


def test_Shape(capsys):
    input_op = Input((1, 3, 2, 2), np.dtype(np.float32))
    shape_op = Shape(input_op)
    PrintVisitor().visit(shape_op)
    captured = capsys.readouterr()
    expected_output = """\
Input_0                         : Input((1, 3, 2, 2), dtype=float32)
Shape_0                         : Shape(Input_0)
"""
    assert captured.out == expected_output


def test_Sigmoid(capsys):
    input_op = Input((1, 3, 2, 2), np.dtype(np.float32))
    sigmoid_op = Sigmoid(input_op)
    PrintVisitor().visit(sigmoid_op)
    captured = capsys.readouterr()
    expected_output = """\
Input_0                         : Input((1, 3, 2, 2), dtype=float32)
Sigmoid_0                       : Sigmoid(Input_0)
"""
    assert captured.out == expected_output


def test_Sign(capsys):
    input_op = Input((1, 5), np.dtype(np.float32))
    sign_op = Sign(input_op)
    PrintVisitor().visit(sign_op)
    captured = capsys.readouterr()
    expected_output = """\
Input_0                         : Input((1, 5), dtype=float32)
Sign_0                          : Sign(Input_0)
"""
    assert captured.out == expected_output


def test_Softmax(capsys):
    input_op = Input((1, 5), np.dtype(np.float32))
    softmax_op = Softmax(input_op)
    PrintVisitor().visit(softmax_op)
    captured = capsys.readouterr()
    expected_output = """\
Input_0                         : Input((1, 5), dtype=float32)
Softmax_0                       : Softmax(Input_0, axis=-1)
"""
    assert captured.out == expected_output

    input_op = Input((1, 5), np.dtype(np.float32))
    softmax_op = Softmax(input_op, axis=0)
    PrintVisitor().visit(softmax_op)
    captured = capsys.readouterr()
    expected_output = """\
Input_0                         : Input((1, 5), dtype=float32)
Softmax_0                       : Softmax(Input_0, axis=0)
"""
    assert captured.out == expected_output


def test_Sub(capsys):
    input_op = Input((1, 5), np.dtype(np.float32))
    sub_op = Sub(input_op, np.float32(2))
    PrintVisitor().visit(sub_op)
    captured = capsys.readouterr()
    expected_output = """\
Input_0                         : Input((1, 5), dtype=float32)
Sub_0                           : Sub(Input_0, float32_0)
"""
    assert captured.out == expected_output

    input_op = Input((1, 5), np.dtype(np.float32))
    sub_op = Sub(input_op, input_op)
    PrintVisitor().visit(sub_op)
    captured = capsys.readouterr()
    expected_output = """\
Input_0                         : Input((1, 5), dtype=float32)
Sub_0                           : Sub(Input_0, Input_0)
"""
    assert captured.out == expected_output


def test_Tanh(capsys):
    input_op = Input((1, 5), np.dtype(np.float32))
    tanh_op = Tanh(input_op)
    PrintVisitor().visit(tanh_op)
    captured = capsys.readouterr()
    expected_output = """\
Input_0                         : Input((1, 5), dtype=float32)
Tanh_0                          : Tanh(Input_0)
"""
    assert captured.out == expected_output


def test_Tile(capsys):
    input_op = Input((1, 5), np.dtype(np.float32))
    tile_op = Tile(input_op, np.array([1, 2]))
    PrintVisitor().visit(tile_op)
    captured = capsys.readouterr()
    expected_output = """\
Input_0                         : Input((1, 5), dtype=float32)
Tile_0                          : Tile(Input_0, [1 2])
"""
    assert captured.out == expected_output


def test_Transpose(capsys):
    input_op = Input((1, 5), np.dtype(np.float32))
    transpose_op = Transpose(input_op, permutation=np.array([1, 0]))
    PrintVisitor().visit(transpose_op)
    captured = capsys.readouterr()
    expected_output = """\
Input_0                         : Input((1, 5), dtype=float32)
Transpose_0                     : Transpose(Input_0, permutation=[1 0])
"""
    assert captured.out == expected_output

    input_op = Input((1, 3, 2, 2), np.dtype(np.float32))
    transpose_op = Transpose(input_op, permutation=np.array([0, 2, 3, 1]))
    PrintVisitor().visit(transpose_op)
    captured = capsys.readouterr()
    expected_output = """\
Input_0                         : Input((1, 3, 2, 2), dtype=float32)
Transpose_0                     : Transpose(Input_0, permutation=[0 2 3 1])
"""
    assert captured.out == expected_output


def test_Unsqueeze(capsys):
    input_op = Input((1, 5), np.dtype(np.float32))
    unsqueeze_op = Unsqueeze(input_op, axes=np.array([0]))
    PrintVisitor().visit(unsqueeze_op)
    captured = capsys.readouterr()
    expected_output = """\
Input_0                         : Input((1, 5), dtype=float32)
Unsqueeze_0                     : Unsqueeze(Input_0, axes=[0])
"""
    assert captured.out == expected_output
