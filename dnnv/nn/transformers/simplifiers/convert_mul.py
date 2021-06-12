import numpy as np

from typing import Union

from .base import Simplifier
from ... import operations
from ...graph import OperationGraph


class ConvertMul(Simplifier):
    def visit_Mul(
        self, operation: operations.Mul
    ) -> Union[operations.Mul, operations.Gemm]:
        a = operation.a
        b = operation.b
        if isinstance(a, operations.Operation) and isinstance(b, operations.Operation):
            return operation
        elif isinstance(a, operations.Operation):
            input_op = a
            c = b
            if np.all(c == 1):
                return input_op
            elif np.all(c == 0):
                input_shape = OperationGraph([input_op]).output_shape[0]
                output_shape = np.broadcast(np.zeros(input_shape), c).shape
                return np.zeros_like(c)
        elif isinstance(b, operations.Operation):
            input_op = b
            c = a
            if np.all(c == 1):
                return input_op
            elif np.all(c == 0):
                input_shape = OperationGraph([input_op]).output_shape[0]
                output_shape = np.broadcast(np.zeros(input_shape), c).shape
                return np.zeros_like(c)
        else:
            return a * b
        input_shape = OperationGraph([input_op]).output_shape[0]
        output_shape = np.broadcast(np.zeros(input_shape), c).shape
        if output_shape != input_shape:
            return operation
        b = np.zeros_like(np.reshape(c, -1))
        if len(input_shape) == 0:
            return operation
        elif len(input_shape) == 1:
            w = np.diag(np.reshape(c, -1))
        elif len(input_shape) == 2:
            if np.size(c) > input_shape[-1]:
                return operation
            w = np.diag(np.reshape(c, -1))
        elif len(input_shape) == 4:
            num_channels = input_shape[1]
            if not np.size(c) == num_channels:
                return operation
            w = np.diag(np.reshape(c, -1)).reshape(num_channels, num_channels, 1, 1)
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
        return operations.Add(operations.MatMul(input_op, w), b)


__all__ = ["ConvertMul"]
