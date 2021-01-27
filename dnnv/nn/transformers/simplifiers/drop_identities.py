from .base import Simplifier
from ... import operations
from ...graph import OperationGraph


class DropIdentity(Simplifier):
    def visit_Identity(self, operation: operations.Identity):
        return operation.x


class DropUnnecessaryConcat(Simplifier):
    def visit_Concat(self, operation: operations.Concat) -> operations.Operation:
        if len(operation.x) == 1:
            return operation.x[0]
        return operation


class DropUnnecessaryFlatten(Simplifier):
    def visit_Flatten(self, operation: operations.Flatten) -> operations.Operation:
        if (
            operation.axis == 1
            and len(OperationGraph([operation.x]).output_shape[0]) == 2
        ):
            return operation.x
        return operation
