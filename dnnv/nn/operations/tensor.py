import numpy as np

from .base import Operation
from ..utils import as_numpy


class Cast(Operation):
    def __init__(self, x, to):
        self.x = x
        self.to = to

    @classmethod
    def from_onnx(cls, onnx_node, *inputs):
        attributes = {a.name: as_numpy(a) for a in onnx_node.attribute}
        axis = attributes.get("to")
        return cls(inputs, axis=axis)


class Concat(Operation):
    def __init__(self, x, axis):
        self.x = x
        self.axis = axis

    @classmethod
    def from_onnx(cls, onnx_node, *inputs):
        attributes = {a.name: as_numpy(a) for a in onnx_node.attribute}
        axis = attributes.get("axis")
        return cls(inputs, axis=axis)


class Expand(Operation):
    def __init__(self, x, shape):
        self.x = x
        self.shape = shape

    @classmethod
    def from_onnx(cls, onnx_node, *inputs):
        return cls(*inputs)


class Flatten(Operation):
    def __init__(self, x, *, axis=1):
        self.x = x
        self.axis = axis

    @classmethod
    def from_onnx(cls, onnx_node, *inputs):
        attributes = {a.name: as_numpy(a) for a in onnx_node.attribute}
        axis = attributes.get("axis", 1)
        return cls(*inputs, axis=axis)


class Gather(Operation):
    def __init__(self, x, indices, *, axis=0):
        self.x = x
        self.indices = indices
        self.axis = axis

    @classmethod
    def from_onnx(cls, onnx_node, *inputs):
        attributes = {a.name: as_numpy(a) for a in onnx_node.attribute}
        axis = attributes.get("axis", 0)
        return cls(*inputs, axis=axis)


class Identity(Operation):
    def __init__(self, x):
        self.x = x

    @classmethod
    def from_onnx(cls, onnx_node, *inputs):
        return cls(*inputs)


class Pad(Operation):
    def __init__(self, x, pads, *, mode="constant", value=0.0):
        self.x = x
        self.pads = pads
        self.mode = mode
        self.value = value

    @classmethod
    def from_onnx(cls, onnx_node, *inputs):
        attributes = {a.name: as_numpy(a) for a in onnx_node.attribute}
        mode = attributes.get("mode", "constant")
        pads = attributes.get("pads")
        value = attributes.get("value", 0.0)
        return cls(*inputs, pads, mode=mode, value=value)


class Reshape(Operation):
    def __init__(self, x, shape):
        self.x = x
        self.shape = shape

    @classmethod
    def from_onnx(cls, onnx_node, *inputs):
        return cls(*inputs)


class Shape(Operation):
    def __init__(self, x):
        self.x = x

    @classmethod
    def from_onnx(cls, onnx_node, *inputs):
        return cls(*inputs)


class Tile(Operation):
    def __init__(self, x, repeats):
        self.x = x
        self.repeats = repeats

    @classmethod
    def from_onnx(cls, onnx_node, *inputs):
        return cls(*inputs)


class Transpose(Operation):
    def __init__(self, x, *, permutation=None):
        self.x = x
        if permutation is not None:
            self.permutation = permutation
        else:
            self.permutation = np.arange(len(self.x.shape) - 1, -1, -1)

    @classmethod
    def from_onnx(cls, onnx_node, *inputs):
        attributes = {a.name: as_numpy(a) for a in onnx_node.attribute}
        perm = attributes.get("perm")
        return cls(*inputs, permutation=perm)


class Unsqueeze(Operation):
    def __init__(self, x, axes):
        self.x = x
        self.axes = axes

    @classmethod
    def from_onnx(cls, onnx_node, *inputs):
        attributes = {a.name: as_numpy(a) for a in onnx_node.attribute}
        axes = attributes.get("axes")
        if isinstance(inputs[0], np.ndarray):
            a = inputs[0]
            for axis in axes:
                a = np.expand_dims(a, axis)
            return a
        return cls(*inputs, axes=axes)
