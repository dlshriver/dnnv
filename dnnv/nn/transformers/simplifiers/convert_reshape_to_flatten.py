from ... import operations
from .base import Simplifier


class ConvertReshapeToFlatten(Simplifier):
    def visit_Reshape(self, operation: operations.Reshape):
        if not isinstance(operation.shape, operations.Concat):
            return operation
        concat: operations.Concat = operation.shape
        if concat.axis != 0:
            return operation
        if len(concat.x) != 2 or concat.x[1] != -1:
            return operation
        if not isinstance(concat.x[0], operations.Unsqueeze):
            return operation
        unsqueeze: operations.Unsqueeze = concat.x[0]
        if len(unsqueeze.axes) != 1 or unsqueeze.axes[0] != 0:
            return operation
        if not isinstance(unsqueeze.x, operations.Gather):
            return operation
        gather: operations.Gather = unsqueeze.x
        if gather.axis != 0 or gather.indices.shape != () or gather.indices != 0:
            return operation
        if not isinstance(gather.x, operations.Shape):
            return operation
        shape: operations.Shape = gather.x
        return operations.Flatten(shape.x, axis=1)


__all__ = ["ConvertReshapeToFlatten"]
