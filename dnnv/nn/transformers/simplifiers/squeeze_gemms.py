from typing import Optional

import numpy as np

from ... import operations
from ...graph import OperationGraph
from .base import Simplifier


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
        elif (
            isinstance(operation.a, operations.Flatten)
            and operation.a.axis == 1
            and isinstance(operation.a.x, operations.Conv)
        ):
            if operation.transpose_a:
                return operation
            flatten_op = operation.a
            conv_op = flatten_op.x
            conv_op_graph = OperationGraph([conv_op])[-1:]
            conv_output_shape = conv_op_graph.output_shape[0]
            conv_input_shape = conv_op_graph.input_shape[0]
            dtype = operation.b.dtype
            flat_output_shape = np.product(conv_output_shape[1:])
            if conv_output_shape[1:] == conv_input_shape[1:]:
                W = np.zeros((flat_output_shape, flat_output_shape), dtype=dtype)
                for (b, i, h, w) in np.ndindex(*conv_output_shape):
                    for j in range(conv_input_shape[1]):
                        k = np.ravel_multi_index((b, i, h, w), conv_output_shape)
                        l = np.ravel_multi_index((b, j, h, w), conv_output_shape)
                        W[l, k] = conv_op.w[i, j, 0, 0]
            else:
                # TODO : handle this case
                return operation
            op_b = operation.b.T if operation.transpose_b else operation.b
            W = W @ op_b
            bias: Optional[np.ndarray] = np.array(0, dtype=dtype)
            if conv_op.b is None and operation.c is None:
                bias = None
            if conv_op.b is not None:
                bias = np.zeros(flat_output_shape, dtype=dtype)
                for (b, i, h, w) in np.ndindex(*conv_output_shape):
                    k = np.ravel_multi_index((b, i, h, w), conv_output_shape)
                    bias[k] = conv_op.b[i]
                bias = bias @ op_b
            if operation.c is not None:
                bias = bias + operation.c
            new_flatten_op = operations.Flatten(conv_op.x, axis=flatten_op.axis)
            gemm_op = operations.Gemm(new_flatten_op, W, bias)
            return gemm_op
        return operation


__all__ = ["SqueezeGemms"]
