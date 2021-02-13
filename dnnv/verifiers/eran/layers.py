import numpy as np
import tensorflow.compat.v1 as tf

from typing import Optional, Type, Union

from dnnv import logging
from dnnv.nn.layers import Layer, Convolutional
from dnnv.nn.operations import (
    Add,
    BatchNormalization,
    Conv,
    Input,
    MaxPool,
    Relu,
    Sigmoid,
    Operation,
    OperationPattern,
)

from .errors import ERANTranslatorError


class _ERANLayerBase(Layer):
    OP_PATTERN: Optional[Union[Type[Operation], OperationPattern]] = None

    def as_tf(self, input_layer):
        raise ERANTranslatorError(
            f"Layer type {self.__class__.__name__} not yet implemented"
        )


class MaxPoolLayer(_ERANLayerBase):
    OP_PATTERN = MaxPool

    def __init__(self, kernel_shape, strides=1, pads=0):
        self.kernel_shape = kernel_shape
        self.strides = strides
        self.pads = pads

    @classmethod
    def from_operation_graph(cls, operation_graph):
        op = operation_graph.output_operations
        assert len(op) == 1
        op = op[0]
        if not isinstance(op, MaxPool):
            raise ValueError(
                f"Expected operation of type MaxPool, but got {op.__class__.__name__}"
            )
        return cls(op.kernel_shape, op.strides, op.pads)

    def as_tf(self, input_layer):
        padding = "SAME"
        if all(p == 0 for p in self.pads):
            padding = "VALID"
        else:
            _, in_height, in_width, _ = [int(d) for d in input_layer.shape]
            out_height = np.ceil(float(in_height) / float(self.strides[0]))
            out_width = np.ceil(float(in_width) / float(self.strides[1]))

            pad_along_height = max(
                (out_height - 1) * self.strides[0] + self.kernel_shape[0] - in_height, 0
            )
            pad_along_width = max(
                (out_width - 1) * self.strides[1] + self.kernel_shape[1] - in_width, 0
            )
            pad_top = pad_along_height // 2
            pad_bottom = pad_along_height - pad_top
            pad_left = pad_along_width // 2
            pad_right = pad_along_width - pad_left
            same_pads = [pad_top, pad_left, pad_bottom, pad_right]
            if any(p1 != p2 for p1, p2 in zip(self.pads, same_pads)):
                raise ERANTranslatorError("Only SAME or VALID padding is not supported")
        maxpool_layer = tf.nn.max_pool(
            input_layer, self.kernel_shape, strides=self.strides, padding=padding
        )
        return maxpool_layer


def build_conv_layer(op):
    activation = None
    if isinstance(op, (Relu, Sigmoid)):
        activation = op.__class__.__name__.lower()
        op = op.inputs
        assert len(op) == 1
        op = op[0]
    elif not isinstance(op, (Conv, BatchNormalization)):
        raise ValueError(
            "Expected operation of type (Conv | BatchNormalization | Activation), but got %s"
            % op.__class__.__name__
        )

    weights = None
    bias = None
    kernel_shape = None
    pads = None
    strides = None
    if isinstance(op, Conv):
        if np.any(op.dilations != 1):
            raise ValueError(
                "Dilation is currently not supported in Convolutional layers."
            )
        if op.group != 1:
            raise ValueError(
                "Grouping is currently not supported in Convolutional layers."
            )
        if not isinstance(op.x, Operation):
            raise ValueError(
                "Constant input tensors are not supported for Conv "
                "in Convolutional layers."
            )
        weights = op.w
        bias = op.b
        kernel_shape = op.kernel_shape
        strides = op.strides
        pads = op.pads
    elif isinstance(op, BatchNormalization):
        std = np.sqrt(op.variance + op.epsilon)
        a = op.scale / std
        b = op.bias - op.scale * op.mean / std

        weights = np.eye(b.shape[0])[:, :, None, None] * a[:, None, None, None]
        bias = b
        kernel_shape = np.array([1, 1])
        pads = (0, 0, 0, 0)
        strides = (1, 1)
    else:
        raise ValueError("Expected type Conv, but got %s" % op.__class__.__name__)

    op = op.inputs
    assert len(op) == 1
    op = op[0]
    return (
        Convolutional(
            weights,
            bias,
            activation=activation,
            kernel_shape=kernel_shape,
            strides=strides,
            pads=pads,
        ),
        op,
    )


def conv_as_tf(conv_layer, x):
    padding = "SAME"
    if isinstance(conv_layer.strides, (int, float)):
        s_h = s_w = float(conv_layer.strides)
    elif len(conv_layer.strides) == 2:
        s_h, s_w = conv_layer.strides
    elif len(conv_layer.strides) == 4:
        s_h, s_w = conv_layer.strides[2:]
    else:
        assert (
            False
        ), f"Unexpected stride configuration: {conv_layer.strides}"  # TODO: clean this up
    if all(p == 0 for p in conv_layer.pads):
        padding = "VALID"
    else:
        _, in_height, in_width, _ = [int(d) for d in x.shape]

        out_height = np.ceil(float(in_height) / float(s_h))
        out_width = np.ceil(float(in_width) / float(s_w))

        pad_along_height = max(
            (out_height - 1) * s_h + conv_layer.kernel_shape[0] - in_height, 0
        )
        pad_along_width = max(
            (out_width - 1) * s_w + conv_layer.kernel_shape[1] - in_width, 0
        )
        pad_top = pad_along_height // 2
        pad_bottom = pad_along_height - pad_top
        pad_left = pad_along_width // 2
        pad_right = pad_along_width - pad_left
        same_pads = [pad_top, pad_left, pad_bottom, pad_right]
        if any(p1 != p2 for p1, p2 in zip(conv_layer.pads, same_pads)):
            raise ERANTranslatorError("Only SAME or VALID padding is supported")
    x = tf.nn.bias_add(
        tf.nn.conv2d(
            x, conv_layer.weights.transpose((2, 3, 1, 0)), (s_h, s_w), padding=padding
        ),
        conv_layer.bias,
    )
    if conv_layer.activation == "relu":
        x = tf.nn.relu(x)
    elif conv_layer.activation == "sigmoid":
        x = tf.nn.sigmoid(x)
    elif conv_layer.activation == "tanh":
        x = tf.nn.tanh(x)
    elif conv_layer.activation is not None:
        raise ERANTranslatorError(
            f"{conv_layer.activation} activation is currently unsupported"
        )
    return x


class Residual(_ERANLayerBase):
    OP_PATTERN = (
        Conv & ((BatchNormalization | Conv) >> Relu >> Conv >> Relu >> Conv)
    ) | (
        ((BatchNormalization | Conv) >> Relu >> Conv >> Relu >> Conv) & Conv
    ) >> Add >> (
        Relu | None
    )

    def __init__(self, conv1, conv2, conv3, downsample, final_activation=None):
        self.conv1 = conv1
        self.conv2 = conv2
        self.conv3 = conv3
        self.downsample = downsample
        self.activation = final_activation

    @classmethod
    def from_operation_graph(cls, operation_graph):
        op = operation_graph.output_operations
        assert len(op) == 1
        op = op[0]
        activation = None
        if isinstance(op, Relu):
            activation = "relu"
            op = op.inputs
            assert len(op) == 1
            op = op[0]
        elif not isinstance(op, Add):
            raise ValueError(
                f"Expected operation of type (Relu | Add), but got {op.__class__.__name__}"
            )

        if isinstance(op, Add):
            op = op.inputs
            assert len(op) == 2
            if isinstance(op[0].inputs[0], Input):
                downsample_op = op[0]
                op = op[1]
            elif isinstance(op[1].inputs[0], Input):
                downsample_op = op[1]
                op = op[0]
            else:
                raise ValueError("Expected one input to Add to be a downsample Conv")
        else:
            raise ValueError(
                f"Expected operation of type Add, but got {op.__class__.__name__}"
            )

        downsample, downsample_op = build_conv_layer(downsample_op)
        assert isinstance(downsample_op, Input)

        conv3, op = build_conv_layer(op)
        conv2, op = build_conv_layer(op)
        conv1, op = build_conv_layer(op)
        assert isinstance(op, Input)

        return cls(conv1, conv2, conv3, downsample, activation)

    def as_tf(self, input_layer):
        downsample_layer = conv_as_tf(self.downsample, input_layer)

        conv1_layer = conv_as_tf(self.conv1, input_layer)
        conv2_layer = conv_as_tf(self.conv2, conv1_layer)
        conv3_layer = conv_as_tf(self.conv3, conv2_layer)

        add_layer = tf.add(downsample_layer, conv3_layer)
        if self.activation == "relu":
            add_layer = tf.nn.relu(add_layer)
        elif self.activation is not None:
            raise ERANTranslatorError(
                f"{self.activation} activation is not currently supported for residual layers"
            )
        return add_layer


ERAN_LAYER_TYPES = [MaxPoolLayer, Residual]
