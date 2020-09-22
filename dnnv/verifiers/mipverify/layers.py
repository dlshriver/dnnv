import numpy as np

from typing import Optional, Type, Union

from dnnv.nn.layers import Layer
from dnnv.nn.operations import MaxPool, Operation, OperationPattern


from .errors import MIPVerifyTranslatorError


class _MIPVerifyLayerBase(Layer):
    OP_PATTERN: Optional[Union[Type[Operation], OperationPattern]] = None

    def as_julia(self, *args, **kwargs):
        raise MIPVerifyTranslatorError(
            f"Layer type {self.__class__.__name__} not yet implemented"
        )


class MaxPoolLayer(_MIPVerifyLayerBase):
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

    def as_julia(self, name, shape, translator_error=MIPVerifyTranslatorError):
        if any(p != 0 for p in self.pads):
            raise translator_error("Padded max pooling is not supported.")
        if any(k != s for k, s in zip(self.kernel_shape, self.strides)):
            raise translator_error("Max pool stride must be equal to kernel size.")
        s_h, s_w = self.strides
        shape[1] = shape[1] // s_h
        shape[2] = shape[2] // s_w
        yield f"{name} = MaxPool((1, {s_h}, {s_w}, 1))"


MIPVERIFY_LAYER_TYPES = [MaxPoolLayer]
