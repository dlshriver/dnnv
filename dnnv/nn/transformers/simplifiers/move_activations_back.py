from typing import Union

from ... import operations
from .base import Simplifier


class MoveActivationsBackward(Simplifier):
    ShapingOperation = (
        operations.Expand
        | operations.Flatten
        | operations.Reshape
        | operations.Transpose
        | operations.Unsqueeze
    )

    def move_back(self, operation: Union[operations.Relu, operations.Sigmoid]):
        if next(self.ShapingOperation.match([operation.x]), None) is None:
            return operation
        # TODO : ensure graph linearity
        output_op = operation.x
        input_op = operation.x.x
        next_op = operation
        while next(self.ShapingOperation.match([input_op]), None) is not None:
            next_op = input_op
            input_op = input_op.x
        operation.x = input_op
        next_op.x = operation
        return output_op

    def visit_Elu(self, operation: operations.Elu):
        return self.move_back(operation)

    def visit_LeakyRelu(self, operation: operations.LeakyRelu):
        return self.move_back(operation)

    def visit_Relu(self, operation: operations.Relu):
        return self.move_back(operation)

    def visit_Sigmoid(self, operation: operations.Sigmoid):
        return self.move_back(operation)

    def visit_Tanh(self, operation: operations.Tanh):
        return self.move_back(operation)
