from typing import Union

from ... import operations
from .base import Simplifier


class ConvertSubToAdd(Simplifier):
    def visit_Sub(
        self, operation: operations.Sub
    ) -> Union[operations.Sub, operations.Add]:
        if isinstance(operation.a, operations.Operation) and isinstance(
            operation.b, operations.Operation
        ):
            return operation
        elif isinstance(operation.a, operations.Operation):
            input_op = operation.a
            c = operation.b
        elif isinstance(operation.b, operations.Operation):
            input_op = operation.b
            c = operation.a
        else:
            return operation
        return operations.Add(input_op, -c)


__all__ = ["ConvertSubToAdd"]
