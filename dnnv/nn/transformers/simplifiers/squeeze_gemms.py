import numpy as np

from .base import Simplifier
from ... import operations


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
            c = np.matmul(input_op.c, b_1) + operation.c
            return operations.Gemm(
                a,
                b,
                c,
                transpose_a=input_op.transpose_a,
                alpha=operation.alpha,
                beta=operation.beta,
            )
        # TODO : reduce when operation.b is Gemm
        return operation
