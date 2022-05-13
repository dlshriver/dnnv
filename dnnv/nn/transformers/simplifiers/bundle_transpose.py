from copy import copy

import numpy as np

from ... import OperationGraph, operations
from .base import Simplifier


class BundleTranspose(Simplifier):
    def visit_Gemm(self, operation: operations.Gemm) -> operations.Gemm:
        if not all(len(s) > 1 for s in OperationGraph([operation]).input_shape):
            return operation
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
            reshape_input_shape = np.asarray(
                OperationGraph([input_op.x]).output_shape[0]
            )
            flat_shape = np.product(reshape_input_shape[1:])
            if flat_shape != input_op.shape[1] or (
                input_op.shape[0] != -1 and input_op.shape[0] != reshape_input_shape[0]
            ):
                return operation
        elif isinstance(input_op, operations.Flatten) and not input_op.axis == 1:
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

        flatten_operation = operations.Flatten(transpose_input_op, axis=1)

        b = operation.b
        if operation.transpose_b:
            b = b.T
        b = b[weights_permutation]

        operation = operations.Gemm(
            flatten_operation,
            b,
            operation.c,
            alpha=operation.alpha,
            beta=operation.beta,
            transpose_a=operation.transpose_a,
            transpose_b=False,
        )
        return operation


__all__ = ["BundleTranspose"]
