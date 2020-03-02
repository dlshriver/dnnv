import numpy as np

from typing import Generator, List, Optional, Tuple, Type, Union

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

from .errors import PlanetTranslatorError


def conv_as_rlv(
    layer: Convolutional,
    layer_id: str,
    prev_layer: Tuple[str, ...],
    input_shape: Tuple[int, int, int, int],
    curr_layer: List[str],
    output_shape: List[int],
) -> Generator[str, None, None]:
    activation = "Linear" if layer.activation is None else "ReLU"
    prev_layer_arr = np.array(prev_layer).reshape(input_shape)

    k_h, k_w = layer.kernel_shape
    p_top, p_left, p_bottom, p_right = layer.pads
    if isinstance(layer.strides, (int, float)):
        s_h = s_w = float(layer.strides)
    elif len(layer.strides) == 2:
        s_h, s_w = layer.strides
    elif len(layer.strides) == 4:
        s_h, s_w = layer.strides[2:]
    else:
        assert (
            False
        ), f"Unexpected stride configuration: {layer.strides}"  # TODO: clean this up

    n, in_c, in_h, in_w = input_shape
    out_c = layer.weights.shape[0]
    out_h = int(np.floor(float(in_h - k_h + p_top + p_bottom) / s_h + 1))
    out_w = int(np.floor(float(in_w - k_w + p_left + p_right) / s_w + 1))
    output_shape.extend([n, out_c, out_h, out_w])

    for k in range(out_c):
        for h in range(out_h):
            r = h * s_h - p_top
            for w in range(out_w):
                c = w * s_w - p_left
                name = f"layer{layer_id}:conv:{k}:{h}:{w}"
                curr_layer.append(name)
                partial_computations = []
                for z in range(in_c):
                    for x in range(k_h):
                        for y in range(k_w):
                            if r + x < 0 or r + x >= in_h:
                                continue
                            if c + y < 0 or c + y >= in_w:
                                continue
                            weight = layer.weights[k, z, x, y]
                            in_name = prev_layer_arr[0, z, r + x, c + y]
                            partial_computations.append(f"{weight:.12f} {in_name}")
                computation = " ".join(partial_computations)
                yield f"{activation} {name} {layer.bias[k]:.12f} {computation}"


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


class _PlanetLayerBase(Layer):
    OP_PATTERN: Optional[Union[Type[Operation], OperationPattern]] = None

    def as_rlv(
        self,
        layer_id: str,
        prev_layer: Tuple[str, ...],
        input_shape: Tuple[int, int, int, int],
        curr_layer: List[str],
        output_shape: List[int],
    ) -> Generator[str, None, None]:
        raise PlanetTranslatorError(
            f"Layer type {self.__class__.__name__} not yet implemented"
        )


class MaxPoolLayer(_PlanetLayerBase):
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

    def as_rlv(
        self,
        layer_id: str,
        prev_layer: Tuple[str, ...],
        input_shape: Tuple[int, int, int, int],
        curr_layer: List[str],
        output_shape: List[int],
    ) -> Generator[str, None, None]:
        prev_layer_arr = np.array(prev_layer).reshape(input_shape)

        k_h, k_w = self.kernel_shape
        p_top, p_left, p_bottom, p_right = self.pads
        s_h, s_w = self.strides

        n, in_c, in_h, in_w = input_shape
        out_c = in_c
        out_h = int(np.floor(float(in_h - k_h + p_top + p_bottom) / s_h + 1))
        out_w = int(np.floor(float(in_w - k_w + p_left + p_right) / s_w + 1))
        output_shape.extend([n, out_c, out_h, out_w])

        for k in range(out_c):
            for h in range(out_h):
                r = h * s_h - p_top
                for w in range(out_w):
                    c = w * s_w - p_left
                    name = f"layer{layer_id}:mp:{k}:{h}:{w}"
                    curr_layer.append(name)
                    partial_computations = []
                    for x in range(k_h):
                        for y in range(k_w):
                            if r + x < 0 or r + x >= in_h:
                                continue
                            if c + y < 0 or c + y >= in_w:
                                continue
                            in_name = prev_layer_arr[0, k, r + x, c + y]
                            partial_computations.append(f"{in_name}")
                    computation = " ".join(partial_computations)
                    yield f"MaxPool {name} {computation}"


class Residual(_PlanetLayerBase):
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

    def as_rlv(
        self,
        layer_id: str,
        prev_layer: Tuple[str, ...],
        input_shape: Tuple[int, int, int, int],
        curr_layer: List[str],
        output_shape: List[int],
    ) -> Generator[str, None, None]:
        downsample_layer: List[str] = []
        downsample_shape: List[int] = []
        for line in conv_as_rlv(
            self.downsample,
            f"{layer_id}d",
            prev_layer,
            input_shape,
            downsample_layer,
            downsample_shape,
        ):
            yield line

        conv1_layer: List[str] = []
        conv1_shape: List[int] = []
        for line in conv_as_rlv(
            self.conv1,
            f"{layer_id}a",
            prev_layer,
            input_shape,
            conv1_layer,
            conv1_shape,
        ):
            yield line
        conv2_layer: List[str] = []
        conv2_shape: List[int] = []
        for line in conv_as_rlv(
            self.conv2,
            f"{layer_id}b",
            tuple(conv1_layer),
            tuple(conv1_shape),
            conv2_layer,
            conv2_shape,
        ):
            yield line
        conv3_layer: List[str] = []
        conv3_shape: List[int] = []
        for line in conv_as_rlv(
            self.conv3,
            f"{layer_id}c",
            tuple(conv2_layer),
            tuple(conv2_shape),
            conv3_layer,
            conv3_shape,
        ):
            yield line

        activation = "Linear" if self.activation is None else "ReLU"
        for i, (left, right) in enumerate(zip(downsample_layer, conv3_layer)):
            name = f"layer{layer_id}:residual:{i}"
            curr_layer.append(name)
            yield f"{activation} {name} 0.0 1.0 {left} 1.0 {right}"
        output_shape.extend(conv3_shape)


PLANET_LAYER_TYPES = [MaxPoolLayer, Residual]
