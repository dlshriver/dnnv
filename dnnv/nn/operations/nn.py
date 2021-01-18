import numpy as np

from collections import Iterable

from .base import Operation
from ..utils import as_numpy


class AveragePool(Operation):
    def __init__(
        self,
        x,
        kernel_shape,
        *,
        ceil_mode=False,
        count_include_pad=False,
        pads=0,
        strides=1
    ):
        self.x = x
        self.kernel_shape = kernel_shape
        self.ceil_mode = ceil_mode
        self.count_include_pad = count_include_pad
        if isinstance(pads, Iterable):
            self.pads = tuple(pads)
        elif isinstance(pads, int):
            self.pads = (pads,) * 4
        if isinstance(strides, Iterable):
            self.strides = tuple(strides)
        elif isinstance(strides, int):
            self.strides = (strides,) * 4

    @classmethod
    def from_onnx(cls, onnx_node, *inputs):
        attributes = {a.name: as_numpy(a) for a in onnx_node.attribute}
        ceil_mode = bool(attributes.get("ceil_mode", False))
        count_include_pad = bool(attributes.get("count_include_pad", False))
        kernel_shape = attributes.get("kernel_shape")
        pads = attributes.get("pads", 0)
        strides = attributes.get("strides", 1)
        return cls(
            *inputs,
            kernel_shape,
            ceil_mode=ceil_mode,
            count_include_pad=count_include_pad,
            pads=pads,
            strides=strides
        )


class BatchNormalization(Operation):
    def __init__(self, x, scale, bias, mean, variance, *, epsilon=1e-5, momentum=0.9):
        self.x = x
        self.scale = scale
        self.bias = bias
        self.mean = mean
        self.variance = variance
        self.epsilon = epsilon
        self.momentum = momentum

    @classmethod
    def from_onnx(cls, onnx_node, *inputs):
        attributes = {a.name: as_numpy(a) for a in onnx_node.attribute}
        epsilon = attributes.get("epsilon", 1e-5)
        momentum = attributes.get("momentum", 0.9)
        return cls(*inputs, epsilon=epsilon, momentum=momentum)


class Conv(Operation):
    def __init__(
        self,
        x,
        w,
        b=None,
        *,
        dilations=(1, 1),
        group=1,
        kernel_shape=None,
        pads=0,
        strides=1
    ):
        self.x = x
        self.w = w
        self.b = b
        self.dilations = np.asarray(dilations)
        self.group = group
        if kernel_shape is not None:
            self.kernel_shape = kernel_shape
        else:
            self.kernel_shape = w.shape[2:]
        if isinstance(pads, Iterable):
            self.pads = tuple(pads)
        elif isinstance(pads, int):
            self.pads = (pads,) * 4
        if isinstance(strides, Iterable):
            self.strides = tuple(strides)
        elif isinstance(strides, int):
            self.strides = (strides,) * 4

    @classmethod
    def from_onnx(cls, onnx_node, *inputs):
        attributes = {a.name: as_numpy(a) for a in onnx_node.attribute}
        dilations = attributes.get("dilations", (1, 1))
        group = attributes.get("group", 1)
        kernel_shape = attributes.get("kernel_shape")
        pads = attributes.get("pads", 0)
        strides = attributes.get("strides", 1)
        return cls(
            *inputs,
            dilations=dilations,
            group=group,
            kernel_shape=kernel_shape,
            pads=pads,
            strides=strides
        )


class Dropout(Operation):
    def __init__(self, x, *, ratio=0.5):
        self.x = x
        self.ratio = ratio

    @classmethod
    def from_onnx(cls, onnx_node, *inputs):
        attributes = {a.name: as_numpy(a) for a in onnx_node.attribute}
        ratio = attributes.get("ratio", 0.5)
        return cls(*inputs, ratio=ratio)


class GlobalAveragePool(Operation):
    def __init__(self, x):
        self.x = x

    @classmethod
    def from_onnx(cls, onnx_node, *inputs):
        return cls(*inputs)


class MaxPool(Operation):
    ROW_MAJOR_STORAGE = 0
    COL_MAJOR_STORAGE = 1

    def __init__(
        self,
        x,
        kernel_shape,
        *,
        ceil_mode=False,
        dilations=(1, 1),
        pads=0,
        storage_order=ROW_MAJOR_STORAGE,
        strides=1
    ):
        self.x = x
        self.ceil_mode = ceil_mode
        self.dilations = np.asarray(dilations)
        self.kernel_shape = kernel_shape
        self.storage_order = storage_order
        if isinstance(pads, Iterable):
            self.pads = tuple(pads)
        elif isinstance(pads, int):
            self.pads = (pads,) * 4
        if isinstance(strides, Iterable):
            self.strides = tuple(strides)
        elif isinstance(strides, int):
            self.strides = (strides,) * 4

    @classmethod
    def from_onnx(cls, onnx_node, *inputs):
        attributes = {a.name: as_numpy(a) for a in onnx_node.attribute}
        ceil_mode = bool(attributes.get("ceil_mode", False))
        dilations = attributes.get("dilations", (1, 1))
        kernel_shape = attributes.get("kernel_shape")
        pads = attributes.get("pads", 0)
        storage_order = attributes.get("storage_order", MaxPool.ROW_MAJOR_STORAGE)
        strides = attributes.get("strides", 1)
        return cls(
            *inputs,
            kernel_shape,
            ceil_mode=ceil_mode,
            dilations=dilations,
            pads=pads,
            storage_order=storage_order,
            strides=strides
        )
