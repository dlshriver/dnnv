import numpy as np

from ... import operations
from .base import Simplifier


class ConvertDivToMul(Simplifier):
    def visit_Div(self, operation: operations.Div) -> operations.Operation:
        if isinstance(operation.a, operations.Operation) and isinstance(
            operation.b, operations.Operation
        ):
            return operation
        elif isinstance(operation.a, operations.Operation):
            input_op = operation.a
            c = operation.b
            if np.all(c == 1):
                return input_op
        elif isinstance(operation.b, operations.Operation):
            c = operation.a
            if np.all(c == 0):
                return c
            return operation
        # TODO : this loses a lot of precision, so don't do it
        # return operations.Mul(input_op, 1.0 / c)
        return operation


__all__ = ["ConvertDivToMul"]
