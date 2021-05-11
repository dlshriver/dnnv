import numpy as np

from copy import copy

from .base import Simplifier
from ... import operations, OperationGraph


class BundleTranspose(Simplifier):
    def visit_Gemm(self, operation: operations.Gemm) -> operations.Gemm:
        if operation.transpose_a:  # TODO : what if operation.b is the Operation?
            return operation
        input_op = operation.a  # TODO : what if operation.b is the Operation?
        if isinstance(input_op, operations.Transpose):
            # TODO : ensure transpose input is flat, and then bundle
            return operation
        if not isinstance(input_op, (operations.Flatten, operations.Reshape)):
            return operation
        flatten_op = input_op
        if isinstance(input_op, operations.Reshape):
            # TODO : check if reshape is a flatten
            return operation
        flatten_input_op = flatten_op.x
        if not isinstance(flatten_input_op, operations.Transpose):
            return operation
        transpose_input_op = flatten_input_op.x

        # TODO : simplify weight permutation computation?
        permutation = np.asarray(flatten_input_op.permutation)
        undo_permutation = permutation[permutation]
        input_shape = np.asarray(OperationGraph([flatten_input_op.x]).output_shape[0])[
            permutation
        ]
        weights_permutation = (
            np.arange(np.product(input_shape))
            .reshape(input_shape)
            .transpose(undo_permutation)
            .flatten()
        )

        flatten_operation = copy(flatten_op)
        flatten_operation.x = transpose_input_op

        operation = copy(operation)
        operation.a = flatten_operation
        b = operation.b
        if operation.transpose_b:
            b = b.T
        operation.b = b[weights_permutation]
        operation.transpose_b = False
        return operation
