from copy import copy

import numpy as np

from ... import operations
from ...analyzers import SplitAnalysis
from .base import Simplifier


class ConvertBatchNorm(Simplifier):
    ANALYSES = {"is_split": SplitAnalysis}

    def visit_BatchNormalization(self, operation: operations.BatchNormalization):
        input_op = operation.x
        if (
            isinstance(input_op, operations.Conv)
            and not self.analysis["is_split"][input_op]
        ):
            std = np.sqrt(operation.variance + operation.epsilon)
            a = operation.scale / std
            b = operation.bias - operation.scale * operation.mean / std

            weights = input_op.w
            a_w = a[:, None, None, None]
            weights = a_w * weights
            bias = input_op.b
            if bias is None:
                bias = np.zeros(weights.shape[0], dtype=weights.dtype)
            bias = a * bias + b

            new_operation = copy(input_op)
            new_operation.w = weights
            new_operation.b = bias
            return new_operation
        elif (
            isinstance(input_op, operations.Gemm)
            and not self.analysis["is_split"][input_op]
        ):
            std = np.sqrt(operation.variance + operation.epsilon)
            a = operation.scale / std
            b = operation.bias - operation.mean * a
            return operations.Gemm(input_op, np.diag(a), b)
        elif isinstance(input_op, operations.Input):
            input_shape = input_op.shape
            input_dtype = input_op.dtype
            if len(input_shape) == 2:
                std = np.sqrt(operation.variance + operation.epsilon)
                a = operation.scale / std
                b = operation.bias - operation.mean * a
                return operations.Gemm(input_op, np.diag(a), b)
            elif len(input_shape) == 4:
                c = operation.mean.shape[0]
                std = np.sqrt(operation.variance + operation.epsilon)
                W = np.zeros(
                    (c, c, 1, 1), dtype=input_dtype
                )  # identity kernel (H, W, inC, outC)
                for i in range(c):
                    W[i, i, 0, 0] = operation.scale[i] / std[i]
                b = operation.bias - operation.scale * operation.mean / std
                op = operations.Conv(input_op, W, b)
                return op
        # TODO : in what other scenarios can BatchNorm be converted?
        return operation


__all__ = ["ConvertBatchNorm"]
