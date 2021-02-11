from .base import Operation
from ..utils import as_numpy


class Add(Operation):
    def __init__(self, a, b):
        self.a = a
        self.b = b

    @classmethod
    def from_onnx(cls, onnx_node, *inputs):
        return cls(*inputs)


class Atan(Operation):
    def __init__(self, x):
        self.x = x

    @classmethod
    def from_onnx(cls, onnx_node, *inputs):
        return cls(*inputs)


class Div(Operation):
    def __init__(self, a, b):
        self.a = a
        self.b = b

    @classmethod
    def from_onnx(cls, onnx_node, *inputs):
        return cls(*inputs)


class Elu(Operation):
    def __init__(self, x, *, alpha=1.0):
        self.x = x
        self.alpha = alpha

    @classmethod
    def from_onnx(cls, onnx_node, *inputs):
        attributes = {a.name: as_numpy(a) for a in onnx_node.attribute}
        alpha = attributes.get("alpha", 1.0)
        return cls(*inputs, alpha=alpha)


class Gemm(Operation):
    def __init__(
        self, a, b, c, *, alpha=1.0, beta=1.0, transpose_a=False, transpose_b=False
    ):
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
            *inputs, alpha=alpha, beta=beta, transpose_a=transA, transpose_b=transB
        )


class LogSoftmax(Operation):
    def __init__(self, x, *, axis=1):
        self.x = x
        self.axis = axis

    @classmethod
    def from_onnx(cls, onnx_node, *inputs):
        attributes = {a.name: as_numpy(a) for a in onnx_node.attribute}
        axis = attributes.get("axis", 1)
        return cls(*inputs, axis=axis)


class MatMul(Operation):
    def __init__(self, a, b):
        self.a = a
        self.b = b

    @classmethod
    def from_onnx(cls, onnx_node, *inputs):
        return cls(*inputs)


class Mul(Operation):
    def __init__(self, a, b):
        self.a = a
        self.b = b

    @classmethod
    def from_onnx(cls, onnx_node, *inputs):
        return cls(*inputs)


class Relu(Operation):
    def __init__(self, x):
        self.x = x

    @classmethod
    def from_onnx(cls, onnx_node, *inputs):
        return cls(*inputs)


class Sigmoid(Operation):
    def __init__(self, x):
        self.x = x

    @classmethod
    def from_onnx(cls, onnx_node, *inputs):
        return cls(*inputs)


class Sign(Operation):
    def __init__(self, x):
        self.x = x

    @classmethod
    def from_onnx(cls, onnx_node, *inputs):
        return cls(*inputs)


class Softmax(Operation):
    def __init__(self, x, *, axis=1):
        self.x = x
        self.axis = axis

    @classmethod
    def from_onnx(cls, onnx_node, *inputs):
        attributes = {a.name: as_numpy(a) for a in onnx_node.attribute}
        axis = attributes.get("axis", 1)
        return cls(*inputs, axis=axis)


class Sub(Operation):
    def __init__(self, a, b):
        self.a = a
        self.b = b

    @classmethod
    def from_onnx(cls, onnx_node, *inputs):
        return cls(*inputs)


class Tanh(Operation):
    def __init__(self, x):
        self.x = x

    @classmethod
    def from_onnx(cls, onnx_node, *inputs):
        return cls(*inputs)
