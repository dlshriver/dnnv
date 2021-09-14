import numpy as np

from .base import Simplifier
from ... import operations
from ...graph import OperationGraph


class SqueezeGemms(Simplifier):
    def visit_Gemm(self, operation: operations.Gemm) -> operations.Gemm:
        if isinstance(operation.a, operations.Gemm) and not operation.transpose_a:
            input_op = operation.a
            if (
                not isinstance(input_op.a, operations.Operation)
                or input_op.alpha != 1.0
                or input_op.beta != 1.0
            ):
                return operation
            a = input_op.a
            b_0 = input_op.b.T if input_op.transpose_b else input_op.b
            b_1 = operation.b.T if operation.transpose_b else operation.b
            b = np.matmul(b_0, b_1)
            if input_op.c is not None and operation.c is not None:
                c = np.matmul(input_op.c, b_1) + operation.c
            elif operation.c is not None:
                c = operation.c
            elif input_op.c is not None:
                c = np.matmul(input_op.c, b_1)
            else:
                c = None
            return operations.Gemm(
                a,
                b,
                c,
                transpose_a=input_op.transpose_a,
                alpha=operation.alpha,
                beta=operation.beta,
            )
        elif isinstance(operation.b, operations.Gemm):
            # TODO : reduce when operation.b is Gemm
            return operation
        elif isinstance(operation.a, operations.Flatten) and isinstance(
            operation.a.x, operations.Conv
        ):
            if operation.transpose_a:
                return operation
            flatten_op = operation.a
            conv_op = flatten_op.x
            if conv_op.w.shape[0] != conv_op.w.shape[1]:
                return operation
            if conv_op.w.shape[2] != conv_op.w.shape[3] and conv_op.shape[2] != 1:
                # TODO : handle this case
                return operation
            input_shape = OperationGraph([conv_op]).output_shape[0]
            flat_input_shape = np.product(input_shape[1:])
            W = np.zeros((flat_input_shape, flat_input_shape)).astype(operation.b.dtype)
            for (b, i, h, w) in np.ndindex(input_shape):
                for j in range(input_shape[1]):
                    k = np.ravel_multi_index((b, i, h, w), input_shape)
                    l = np.ravel_multi_index((b, j, h, w), input_shape)
                    W[k, l] = conv_op.w[i, j, 0, 0]
            op_b = operation.b
            if operation.transpose_b:
                op_b = op_b.T
            W = W @ op_b
            bias = np.tile(conv_op.b, np.product(input_shape[2:]))
            bias = bias @ op_b + operation.c
            new_flatten_op = operations.Flatten(conv_op.x, axis=flatten_op.axis)
            gemm_op = operations.Gemm(new_flatten_op, W, bias)
            return gemm_op
        return operation
