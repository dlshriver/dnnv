import numpy as np

from typing import Union

from .base import Simplifier
from ... import operations
from ...graph import OperationGraph


class ConvertAdd(Simplifier):
    def visit_Add(
        self, operation: operations.Add
    ) -> Union[operations.Add, operations.Gemm]:
        a = operation.a
        b = operation.b
        if isinstance(a, operations.Operation) and isinstance(b, operations.Operation):
            return operation
        elif isinstance(a, operations.Operation):
            input_op = a
            c = b
            if np.all(c == 0):
                return input_op
        elif isinstance(b, operations.Operation):
            input_op = b
            c = a
            if np.all(c == 0):
                return input_op
        else:
            return a + b
        input_shape = OperationGraph([input_op]).output_shape[0]
        if len(input_shape) == 0:
            return operation
            w = np.eye(1, dtype=c.dtype)
        elif len(input_shape) == 1:
            return operation
            w = np.eye(input_shape[0], dtype=c.dtype)
        elif len(input_shape) == 2:
            w = np.eye(input_shape[1], dtype=c.dtype)
        elif len(input_shape) == 4:
            return operation
            num_channels = input_shape[1]
            if not np.size(c) == num_channels:
                return operation
            w = np.eye(num_channels, dtype=c.dtype).reshape(
                num_channels, num_channels, 1, 1
            )
            c = np.reshape(c, -1)
            return operations.Conv(
                input_op,
                w,
                c,
                kernel_shape=np.array([num_channels, num_channels]),
                pads=np.array([0, 0, 0, 0]),
                strides=np.array([1, 1]),
            )
        else:
            return operation
        return operations.Gemm(input_op, w, c)


__all__ = ["ConvertAdd"]
