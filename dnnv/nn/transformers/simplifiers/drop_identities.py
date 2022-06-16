from ... import operations
from ...graph import OperationGraph
from .base import Simplifier


class DropDropout(Simplifier):
    def visit_Dropout(self, operation: operations.Dropout):
        if operation.training_mode:
            return operation
        return operation.x


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


class DropUnnecessaryRelu(Simplifier):
    def visit_Relu(self, operation: operations.Relu):
        input_op = operation.x
        if isinstance(
            input_op, (operations.Relu, operations.Sigmoid, operations.Softmax)
        ):
            return input_op
        return operation


__all__ = [
    "DropDropout",
    "DropIdentity",
    "DropUnnecessaryConcat",
    "DropUnnecessaryFlatten",
    "DropUnnecessaryRelu",
]
