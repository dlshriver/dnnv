import numpy as np

from ... import operations
from ...graph import OperationGraph
from .base import Simplifier


class ConvertMul(Simplifier):
    def visit_Mul(self, operation: operations.Mul) -> operations.Operation:
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
        else:
            return operation
        return operations.Add(operations.MatMul(input_op, w), b)


__all__ = ["ConvertMul"]
