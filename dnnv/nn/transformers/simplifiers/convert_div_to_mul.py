import numpy as np
from typing import Union

from .base import Simplifier
from ... import operations


class ConvertDivToMul(Simplifier):
    def visit_Div(
        self, operation: operations.Div
    ) -> Union[operations.Div, operations.Mul]:
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
        else:
            return operation
        if isinstance(input_op, operations.Add):
            return operation
        # TODO : this loses a lot of precision, so don't do it
        # return operations.Mul(input_op, 1.0 / c)
        return operation


__all__ = ["ConvertDivToMul"]
