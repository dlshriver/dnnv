"""
"""
import numpy as np
import onnx

from typing import Optional, Sequence, Union

from .patterns import Or, Parallel, Sequential
from ..utils import ONNX_TO_NUMPY_DTYPE
from ... import logging
from ...utils import get_subclasses


class Op(type):
    def __str__(self):
        return self.__name__

    def __and__(self, other) -> Parallel:
        return Parallel(self, other)

    def __rand__(self, other) -> Parallel:
        return Parallel(other, self)

    def __or__(self, other) -> Or:
        return Or(self, other)

    def __ror__(self, other) -> Or:
        return Or(other, self)

    def __rshift__(self, other) -> Sequential:
        return Sequential(self, other)

    def __rrshift__(self, other) -> Sequential:
        return Sequential(other, self)


class Operation(metaclass=Op):
    def __getitem__(self, index):
        return OutputSelect(self, index)

    def __hash__(self):
        return hash(type(self))

    def __str__(self):
        return type(self).__name__

    @property
    def inputs(self):
        inputs = []
        for value in self.__dict__.values():
            if isinstance(value, Operation):
                inputs.append(value)
            elif isinstance(value, (list, tuple)):
                for sub_value in value:
                    if isinstance(sub_value, Operation):
                        inputs.append(sub_value)
        return inputs

    @classmethod
    def from_onnx(cls, onnx_node, *inputs):
        logger = logging.getLogger(__name__)
        if isinstance(onnx_node, onnx.ValueInfoProto):
            return Input.from_onnx(onnx_node)
        op_type = onnx_node.op_type
        for op_cls in get_subclasses(cls):
            if op_type == op_cls.__name__:
                operation = op_cls.from_onnx(onnx_node, *inputs)
                if all(not isinstance(i, Operation) for i in inputs) and isinstance(
                    operation, Operation
                ):
                    logger.warning(
                        "Operation on constant inputs returned non-constant."
                    )
                return operation
        raise ValueError("Unimplemented operation type: %s" % op_type)

    @classmethod
    def match(cls, operations: Sequence["Operation"]):
        if len(operations) < 1:
            return None
        operation = operations[0]
        if not isinstance(operation, cls):
            return None
        for op in operations:
            if op is not operation:
                return None
        yield operation.inputs


class Input(Operation):
    def __init__(self, shape, dtype):
        self.shape = shape
        self.dtype = dtype

    @classmethod
    def from_onnx(cls, onnx_node, *inputs):
        dims = [
            -1 if dim.dim_param else int(dim.dim_value)
            for dim in onnx_node.type.tensor_type.shape.dim
        ]
        shape = np.array(dims)
        dtype = ONNX_TO_NUMPY_DTYPE[onnx_node.type.tensor_type.elem_type]
        return cls(shape, dtype=dtype)


class OutputSelect(Operation):
    def __init__(self, operation, index):
        self.operation = operation
        self.index = index
