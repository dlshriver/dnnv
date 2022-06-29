import numpy as np

from typing import Optional

from .base import Operation
from ..utils import as_numpy


class Cast(Operation):
    def __init__(self, x, to, *, name: Optional[str] = None):
        super().__init__(name=name)
        self.x = x
        self.to = to

    @classmethod
    def from_onnx(cls, onnx_node, *inputs):
        attributes = {a.name: as_numpy(a) for a in onnx_node.attribute}
        to = attributes.get("to")
        return cls(*inputs, to=to, name=onnx_node.name)


class Concat(Operation):
    def __init__(self, x, axis, *, name: Optional[str] = None):
        super().__init__(name=name)
        self.x = x
        self.axis = axis

    @classmethod
    def from_onnx(cls, onnx_node, *inputs):
        attributes = {a.name: as_numpy(a) for a in onnx_node.attribute}
        axis = attributes.get("axis")
        return cls(inputs, axis=axis, name=onnx_node.name)


class Expand(Operation):
    def __init__(self, x, shape, *, name: Optional[str] = None):
        super().__init__(name=name)
        self.x = x
        self.shape = shape

    @classmethod
    def from_onnx(cls, onnx_node, *inputs):
        return cls(*inputs, name=onnx_node.name)


class Flatten(Operation):
    def __init__(self, x, *, axis=1, name: Optional[str] = None):
        super().__init__(name=name)
        self.x = x
        self.axis = axis

    @classmethod
    def from_onnx(cls, onnx_node, *inputs):
        attributes = {a.name: as_numpy(a) for a in onnx_node.attribute}
        axis = attributes.get("axis", 1)
        return cls(*inputs, axis=axis, name=onnx_node.name)


class Gather(Operation):
    def __init__(self, x, indices, *, axis=0, name: Optional[str] = None):
        super().__init__(name=name)
        self.x = x
        self.indices = indices
        self.axis = axis

    @classmethod
    def from_onnx(cls, onnx_node, *inputs):
        attributes = {a.name: as_numpy(a) for a in onnx_node.attribute}
        axis = attributes.get("axis", 0)
        return cls(*inputs, axis=axis, name=onnx_node.name)


class Identity(Operation):
    def __init__(self, x, *, name: Optional[str] = None):
        super().__init__(name=name)
        self.x = x

    @classmethod
    def from_onnx(cls, onnx_node, *inputs):
        return cls(*inputs, name=onnx_node.name)


class Pad(Operation):
    def __init__(
        self, x, pads, *, mode="constant", value=0.0, name: Optional[str] = None
    ):
        super().__init__(name=name)
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
        return cls(*inputs, pads, mode=mode, value=value, name=onnx_node.name)


class Reshape(Operation):
    def __init__(self, x, shape, *, allowzero=False, name: Optional[str] = None):
        super().__init__(name=name)
        self.x = x
        self.shape = shape
        self.allowzero = allowzero

    @classmethod
    def from_onnx(cls, onnx_node, *inputs):
        attributes = {a.name: as_numpy(a) for a in onnx_node.attribute}
        allowzero = attributes.get("allowzero", False)
        return cls(*inputs, allowzero=allowzero, name=onnx_node.name)


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
        nearest_mode="round_prefer_floor",
        name: Optional[str] = None
    ):
        super().__init__(name=name)
        # assert scales.size != 0 or sizes.size != 0
        # assert scales.size == 0 or sizes.size == 0
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
            nearest_mode=nearest_mode,
            name=onnx_node.name
        )


class Shape(Operation):
    def __init__(self, x, *, name: Optional[str] = None):
        super().__init__(name=name)
        self.x = x

    @classmethod
    def from_onnx(cls, onnx_node, *inputs):
        return cls(*inputs, name=onnx_node.name)


class Tile(Operation):
    def __init__(self, x, repeats, *, name: Optional[str] = None):
        super().__init__(name=name)
        self.x = x
        self.repeats = repeats

    @classmethod
    def from_onnx(cls, onnx_node, *inputs):
        return cls(*inputs, name=onnx_node.name)


class Transpose(Operation):
    def __init__(self, x, *, permutation=None, name: Optional[str] = None):
        super().__init__(name=name)
        self.x = x
        if permutation is not None:
            self.permutation = permutation
        else:
            self.permutation = np.arange(len(self.x.shape) - 1, -1, -1)

    @classmethod
    def from_onnx(cls, onnx_node, *inputs):
        attributes = {a.name: as_numpy(a) for a in onnx_node.attribute}
        perm = attributes.get("perm")
        return cls(*inputs, permutation=perm, name=onnx_node.name)


class Unsqueeze(Operation):
    def __init__(self, x, axes, *, name: Optional[str] = None):
        super().__init__(name=name)
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
        return cls(*inputs, axes=axes, name=onnx_node.name)


class Split(Operation):
    def __init__(self, x, axis, split, *, name: Optional[str] = None):
        super().__init__(name=name)
        self.x = x
        self.axis = axis
        self.split = split

    @classmethod
    def from_onnx(cls, onnx_node, *inputs):
        attributes = {a.name: as_numpy(a) for a in onnx_node.attribute}
        axis = attributes.get("axis")
        split = attributes.get("split")
        split = tuple(split)
        return cls(*inputs, axis=axis, split=split, name=onnx_node.name)


class ReduceL2(Operation):
    def __init__(self, x, axes, keepdims, *, name: Optional[str] = None):
        super().__init__(name=name)
        self.x = x
        if isinstance(axes, int):
            self.axes = axes
        elif len(axes) == 1:
            self.axes = int(axes[0])
        elif len(axes) > 1:
            self.axes = tuple(axes)
        self.keepdims = keepdims

    @classmethod
    def from_onnx(cls, onnx_node, *inputs):
        attributes = {a.name: as_numpy(a) for a in onnx_node.attribute}
        axes = attributes.get("axes")
        keepdims = attributes.get("keepdims")
        return cls(*inputs, axes=axes, keepdims=keepdims, name=onnx_node.name)


class Clip(Operation):
    def __init__(self, x, min, max, *, name: Optional[str] = None):
        super().__init__(name=name)
        self.x = x
        self.min = min if min is not None else -np.inf
        self.max = max if max is not None else +np.inf

    @classmethod
    def from_onnx(cls, onnx_node, *inputs):
        attributes = {a.name: as_numpy(a) for a in onnx_node.attribute}
        _min = attributes.get("min")
        _max = attributes.get("max")
        return cls(*inputs, min=_min, max=_max, name=onnx_node.name)


class Squeeze(Operation):
    def __init__(self, x, axes, *, name: Optional[str] = None):
        super().__init__(name=name)
        self.x = x
        self.axes = axes

    @classmethod
    def from_onnx(cls, onnx_node, *inputs):
        attributes = {a.name: as_numpy(a) for a in onnx_node.attribute}
        axes = attributes.get("axes")
        return cls(*inputs, axes=axes, name=onnx_node.name)


class Upsample(Operation):
    def __init__(self, x, scales, mode, *, name: Optional[str] = None):
        super().__init__(name=name)
        self.x = x
        self.scales = scales
        self.mode = mode

    @classmethod
    def from_onnx(cls, onnx_node, *inputs):
        attributes = {a.name: as_numpy(a) for a in onnx_node.attribute}
        mode = attributes.get("mode")
        # scales = attributes.get("scales")
        return cls(*inputs, mode=mode, name=onnx_node.name)

class Slice(Operation):
    def __init__(
        self, x, starts, ends, axes=None, steps=None, *, name: Optional[str] = None
    ):
        super().__init__(name=name)
        self.x = x
        self.starts = starts
        self.ends = ends
        self.axes = axes
        self.steps = steps

    @classmethod
    def from_onnx(cls, onnx_node, *inputs):
        attributes = {a.name: as_numpy(a) for a in onnx_node.attribute}
        return cls(*inputs, name=onnx_node.name)

__all__ = [
    "Cast",
    "Concat",
    "Expand",
    "Flatten",
    "Gather",
    "Identity",
    "Pad",
    "Reshape",
    "Resize",
    "Shape",
    "Tile",
    "Transpose",
    "Unsqueeze",
    "Split",
    "ReduceL2",
    "Clip",
    "Squeeze",
    "Upsample",
    "Slice"
]
