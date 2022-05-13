from typing import Optional

from ..utils import as_numpy
from .base import Operation


class Add(Operation):
    def __init__(self, a, b, *, name: Optional[str] = None):
        super().__init__(name=name)
        self.a = a
        self.b = b

    @classmethod
    def from_onnx(cls, onnx_node, *inputs):
        return cls(*inputs, name=onnx_node.name)


class Atan(Operation):
    def __init__(self, x, *, name: Optional[str] = None):
        super().__init__(name=name)
        self.x = x

    @classmethod
    def from_onnx(cls, onnx_node, *inputs):
        return cls(*inputs, name=onnx_node.name)


class Div(Operation):
    def __init__(self, a, b, *, name: Optional[str] = None):
        super().__init__(name=name)
        self.a = a
        self.b = b

    @classmethod
    def from_onnx(cls, onnx_node, *inputs):
        return cls(*inputs, name=onnx_node.name)


class Elu(Operation):
    def __init__(self, x, *, alpha=1.0, name: Optional[str] = None):
        super().__init__(name=name)
        self.x = x
        self.alpha = alpha

    @classmethod
    def from_onnx(cls, onnx_node, *inputs):
        attributes = {a.name: as_numpy(a) for a in onnx_node.attribute}
        alpha = attributes.get("alpha", 1.0)
        return cls(*inputs, alpha=alpha, name=onnx_node.name)


class Gemm(Operation):
    def __init__(
        self,
        a,
        b,
        c=None,
        *,
        alpha=1.0,
        beta=1.0,
        transpose_a=False,
        transpose_b=False,
        name: Optional[str] = None
    ):
        super().__init__(name=name)
        self.a = a
        self.b = b
        self.c = c
        self.alpha = alpha
        self.beta = beta
        self.transpose_a = transpose_a
        self.transpose_b = transpose_b

    @classmethod
    def from_onnx(cls, onnx_node, *inputs):
        attributes = {a.name: as_numpy(a) for a in onnx_node.attribute}
        alpha = attributes.get("alpha", 1.0)
        beta = attributes.get("beta", 1.0)
        transA = bool(attributes.get("transA", False))
        transB = bool(attributes.get("transB", False))
        return cls(
            *inputs,
            alpha=alpha,
            beta=beta,
            transpose_a=transA,
            transpose_b=transB,
            name=onnx_node.name
        )


class LeakyRelu(Operation):
    def __init__(self, x, *, alpha=0.01, name: Optional[str] = None):
        super().__init__(name=name)
        self.x = x
        self.alpha = alpha

    @classmethod
    def from_onnx(cls, onnx_node, *inputs):
        attributes = {a.name: as_numpy(a) for a in onnx_node.attribute}
        alpha = attributes.get("alpha", 0.01)
        return cls(*inputs, alpha=alpha, name=onnx_node.name)


class LogSoftmax(Operation):
    def __init__(self, x, *, axis=-1, name: Optional[str] = None):
        super().__init__(name=name)
        self.x = x
        self.axis = axis

    @classmethod
    def from_onnx(cls, onnx_node, *inputs):
        attributes = {a.name: as_numpy(a) for a in onnx_node.attribute}
        axis = attributes.get("axis", 1)
        return cls(*inputs, axis=axis, name=onnx_node.name)


class MatMul(Operation):
    def __init__(self, a, b, *, name: Optional[str] = None):
        super().__init__(name=name)
        self.a = a
        self.b = b

    @classmethod
    def from_onnx(cls, onnx_node, *inputs):
        return cls(*inputs, name=onnx_node.name)


class Mul(Operation):
    def __init__(self, a, b, *, name: Optional[str] = None):
        super().__init__(name=name)
        self.a = a
        self.b = b

    @classmethod
    def from_onnx(cls, onnx_node, *inputs):
        return cls(*inputs, name=onnx_node.name)


class Relu(Operation):
    def __init__(self, x, *, name: Optional[str] = None):
        super().__init__(name=name)
        self.x = x

    @classmethod
    def from_onnx(cls, onnx_node, *inputs):
        return cls(*inputs, name=onnx_node.name)


class Sigmoid(Operation):
    def __init__(self, x, *, name: Optional[str] = None):
        super().__init__(name=name)
        self.x = x

    @classmethod
    def from_onnx(cls, onnx_node, *inputs):
        return cls(*inputs, name=onnx_node.name)


class Sign(Operation):
    def __init__(self, x, *, name: Optional[str] = None):
        super().__init__(name=name)
        self.x = x

    @classmethod
    def from_onnx(cls, onnx_node, *inputs):
        return cls(*inputs, name=onnx_node.name)


class Softmax(Operation):
    def __init__(self, x, *, axis=-1, name: Optional[str] = None):
        super().__init__(name=name)
        self.x = x
        self.axis = axis

    @classmethod
    def from_onnx(cls, onnx_node, *inputs):
        attributes = {a.name: as_numpy(a) for a in onnx_node.attribute}
        axis = attributes.get("axis", 1)
        return cls(*inputs, axis=axis, name=onnx_node.name)


class Sub(Operation):
    def __init__(self, a, b, name: Optional[str] = None):
        super().__init__(name=name)
        self.a = a
        self.b = b

    @classmethod
    def from_onnx(cls, onnx_node, *inputs):
        return cls(*inputs, name=onnx_node.name)


class Tanh(Operation):
    def __init__(self, x, name: Optional[str] = None):
        super().__init__(name=name)
        self.x = x

    @classmethod
    def from_onnx(cls, onnx_node, *inputs):
        return cls(*inputs, name=onnx_node.name)


__all__ = [
    "Add",
    "Atan",
    "Div",
    "Elu",
    "Gemm",
    "LeakyRelu",
    "LogSoftmax",
    "MatMul",
    "Mul",
    "Relu",
    "Sigmoid",
    "Sign",
    "Softmax",
    "Sub",
    "Tanh",
]
