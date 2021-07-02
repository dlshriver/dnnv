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
    def __init__(self, x, shape, *, allowzero=False):
        self.x = x
        self.shape = shape
        self.allowzero = allowzero

    @classmethod
    def from_onnx(cls, onnx_node, *inputs):
        attributes = {a.name: as_numpy(a) for a in onnx_node.attribute}
        allowzero = attributes.get("allowzero", False)
        return cls(*inputs, allowzero=allowzero)


class Resize(Operation):
    def __init__(
        self,
        x,
        roi,  # np.array([], dtype=np.float32)
        scales,  # np.array([], dtype=np.float)
        sizes,  # np.array([], dtype=np.int64)
        *,
        coordinate_transformation_mode="half_pixel",
        cubic_coeff_a=-0.75,
        exclude_outside=0,
        extrapolation_value=0.0,
        mode="nearest",
        nearest_mode="round_prefer_floor"
    ):
        assert scales.size != 0 or sizes.size != 0
        assert scales.size == 0 or sizes.size == 0
        self.x = x
        self.roi = roi
        self.scales = scales
        self.sizes = sizes
        self.coordinate_transformation_mode = coordinate_transformation_mode
        self.cubic_coeff_a = cubic_coeff_a
        self.exclude_outside = exclude_outside
        self.extrapolation_value = extrapolation_value
        self.mode = mode
        self.nearest_mode = nearest_mode

    @classmethod
    def from_onnx(cls, onnx_node, *inputs):
        attributes = {a.name: as_numpy(a) for a in onnx_node.attribute}
        coordinate_transformation_mode = attributes.get(
            "coordinate_transformation_mode", "half_pixel"
        )
        cubic_coeff_a = attributes.get("cubic_coeff_a", -0.75)
        exclude_outside = attributes.get("exclude_outside", 0)
        extrapolation_value = attributes.get("extrapolation_value", 0.0)
        mode = attributes.get("mode", "nearest")
        nearest_mode = attributes.get("nearest_mode", "round_prefer_floor")
        return cls(
            *inputs,
            coordinate_transformation_mode=coordinate_transformation_mode,
            cubic_coeff_a=cubic_coeff_a,
            exclude_outside=exclude_outside,
            extrapolation_value=extrapolation_value,
            mode=mode,
            nearest_mode=nearest_mode
        )


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
