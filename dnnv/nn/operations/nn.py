import logging
from typing import Optional

import numpy as np

from ..utils import as_numpy
from .base import Operation


class AveragePool(Operation):
    def __init__(
        self,
        x,
        kernel_shape,
        *,
        ceil_mode=False,
        count_include_pad=False,
        pads=None,
        strides=None,
        name: Optional[str] = None,
    ):
        super().__init__(name=name)
        self.x = x
        self.kernel_shape = kernel_shape
        self.ceil_mode = ceil_mode
        self.count_include_pad = count_include_pad

        self.pads = np.asarray(
            pads if pads is not None else (0,) * (len(self.kernel_shape) * 2)
        )
        self.strides = np.asarray(
            strides if strides is not None else (1,) * len(self.kernel_shape)
        )

    @classmethod
    def from_onnx(cls, onnx_node, *inputs):
        attributes = {a.name: as_numpy(a) for a in onnx_node.attribute}
        ceil_mode = bool(attributes.get("ceil_mode", False))
        count_include_pad = bool(attributes.get("count_include_pad", False))
        kernel_shape = attributes.get("kernel_shape")
        pads = attributes.get("pads")
        strides = attributes.get("strides")
        return cls(
            *inputs,
            kernel_shape,
            ceil_mode=ceil_mode,
            count_include_pad=count_include_pad,
            pads=pads,
            strides=strides,
            name=onnx_node.name,
        )


class BatchNormalization(Operation):
    def __init__(
        self,
        x,
        scale,
        bias,
        mean,
        variance,
        *,
        epsilon=1e-5,
        momentum=0.9,
        name: Optional[str] = None,
    ):
        super().__init__(name=name)
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
        return cls(*inputs, epsilon=epsilon, momentum=momentum, name=onnx_node.name)


class Conv(Operation):
    def __init__(
        self,
        x,
        w,
        b=None,
        *,
        dilations=None,
        group=1,
        kernel_shape=None,
        pads=None,
        strides=None,
        name: Optional[str] = None,
    ):
        super().__init__(name=name)
        self.x = x
        self.w = w
        self.b = b

        self.group = group
        self.kernel_shape = np.asarray(
            kernel_shape if kernel_shape is not None else w.shape[2:]
        )
        self.dilations = np.asarray(
            dilations if dilations is not None else (1,) * len(self.kernel_shape)
        )
        self.pads = np.asarray(
            pads if pads is not None else (0,) * (len(self.kernel_shape) * 2)
        )
        self.strides = np.asarray(
            strides if strides is not None else (1,) * len(self.kernel_shape)
        )

    @classmethod
    def from_onnx(cls, onnx_node, *inputs):
        attributes = {a.name: as_numpy(a) for a in onnx_node.attribute}
        dilations = attributes.get("dilations")
        group = attributes.get("group", 1)
        kernel_shape = attributes.get("kernel_shape")
        pads = attributes.get("pads")
        strides = attributes.get("strides")
        return cls(
            *inputs,
            dilations=dilations,
            group=group,
            kernel_shape=kernel_shape,
            pads=pads,
            strides=strides,
            name=onnx_node.name,
        )


class ConvTranspose(Operation):
    def __init__(
        self,
        x,
        w,
        b=None,
        *,
        auto_pad="NOTSET",
        dilations=None,
        group=1,
        kernel_shape=None,
        output_padding=None,
        output_shape=None,
        pads=None,
        strides=None,
        name: Optional[str] = None,
    ):
        super().__init__(name=name)
        self.x = x
        self.w = w
        self.b = b

        self.auto_pad = auto_pad
        self.kernel_shape = np.asarray(
            kernel_shape if kernel_shape is not None else w.shape[2:]
        )
        self.dilations = np.asarray(
            dilations if dilations is not None else (1,) * len(self.kernel_shape)
        )
        self.group = group
        self.output_padding = np.asarray(
            output_padding
            if output_padding is not None
            else (0,) * len(self.kernel_shape)
        )
        self.strides = np.asarray(
            strides if strides is not None else (1,) * len(self.kernel_shape)
        )
        self.output_shape = output_shape
        if self.output_shape is None:
            self.pads = np.asarray(
                pads if pads is not None else (0,) * (len(self.kernel_shape) * 2)
            )
        else:
            raise NotImplementedError(
                "Setting ConvTranspose output_shape is not currently supported"
            )

    @classmethod
    def from_onnx(cls, onnx_node, *inputs):
        attributes = {a.name: as_numpy(a) for a in onnx_node.attribute}
        auto_pad = attributes.get("auto_pad", "NOTSET")
        dilations = attributes.get("dilations")
        group = attributes.get("group", 1)
        kernel_shape = attributes.get("kernel_shape")
        output_padding = attributes.get("output_padding")
        output_shape = attributes.get("output_shape")
        pads = attributes.get("pads")
        strides = attributes.get("strides")
        return cls(
            *inputs,
            auto_pad=auto_pad,
            dilations=dilations,
            group=group,
            kernel_shape=kernel_shape,
            output_padding=output_padding,
            output_shape=output_shape,
            pads=pads,
            strides=strides,
            name=onnx_node.name,
        )


class Dropout(Operation):
    def __init__(
        self,
        x,
        *,
        ratio=0.5,
        training_mode=False,
        include_mask=False,
        name: Optional[str] = None,
    ):
        super().__init__(name=name)
        self.x = x
        self.ratio = ratio
        self.training_mode = training_mode
        self.include_mask = include_mask

    @classmethod
    def from_onnx(cls, onnx_node, *inputs):
        attributes = {a.name: as_numpy(a) for a in onnx_node.attribute}
        ratio = attributes.get("ratio", 0.5)
        training_mode = bool(attributes.get("training_mode", False))
        if training_mode:
            logger = logging.getLogger(__name__)
            logger.warning("Dropout operations in training mode have limited support.")
        if training_mode and len(onnx_node.output) == 2:
            raise NotImplementedError(
                "Using the mask of a Dropout operation is not yet supported."
                " If you need this functionality, please open a GitHub issue."
            )
            return cls(
                *inputs,
                ratio=ratio,
                training_mode=training_mode,
                include_mask=True,
                name=onnx_node.name,
            )
        return cls(
            *inputs, ratio=ratio, training_mode=training_mode, name=onnx_node.name
        )


class GlobalAveragePool(Operation):
    def __init__(self, x, *, name: Optional[str] = None):
        super().__init__(name=name)
        self.x = x

    @classmethod
    def from_onnx(cls, onnx_node, *inputs):
        return cls(*inputs, name=onnx_node.name)


class MaxPool(Operation):
    ROW_MAJOR_STORAGE = 0
    COL_MAJOR_STORAGE = 1

    def __init__(
        self,
        x,
        kernel_shape,
        *,
        ceil_mode=False,
        dilations=None,
        pads=None,
        storage_order=ROW_MAJOR_STORAGE,
        strides=None,
        name: Optional[str] = None,
    ):
        super().__init__(name=name)
        self.x = x
        self.ceil_mode = ceil_mode
        self.storage_order = storage_order

        self.kernel_shape = np.asarray(kernel_shape)
        self.dilations = np.asarray(
            dilations if dilations is not None else (1,) * len(self.kernel_shape)
        )
        self.pads = np.asarray(
            pads if pads is not None else (0,) * (len(self.kernel_shape) * 2)
        )
        self.strides = np.asarray(
            strides if strides is not None else (1,) * len(self.kernel_shape)
        )

    @classmethod
    def from_onnx(cls, onnx_node, *inputs):
        attributes = {a.name: as_numpy(a) for a in onnx_node.attribute}
        ceil_mode = bool(attributes.get("ceil_mode", False))
        dilations = attributes.get("dilations")
        kernel_shape = attributes.get("kernel_shape")
        pads = attributes.get("pads")
        storage_order = attributes.get("storage_order", MaxPool.ROW_MAJOR_STORAGE)
        strides = attributes.get("strides")
        return cls(
            *inputs,
            kernel_shape,
            ceil_mode=ceil_mode,
            dilations=dilations,
            pads=pads,
            storage_order=storage_order,
            strides=strides,
            name=onnx_node.name,
        )


__all__ = [
    "AveragePool",
    "BatchNormalization",
    "Conv",
    "ConvTranspose",
    "Dropout",
    "GlobalAveragePool",
    "MaxPool",
]
