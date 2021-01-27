import numpy as np

from .base import Simplifier
from ... import operations
from ...analyzers import SplitAnalysis


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
                bias = np.zeros(weights.shape[0])
            bias = a * bias + b

            input_op.w = weights
            input_op.b = bias
            return input_op
        elif (
            isinstance(input_op, operations.Gemm)
            and not self.analysis["is_split"][input_op]
        ):
            std = np.sqrt(operation.variance + operation.epsilon)
            a = operation.scale / std
            b = operation.bias - operation.scale * operation.mean / std

            if isinstance(input_op.a, np.ndarray):
                if input_op.transpose_a:
                    input_op.a = np.diag(a) @ input_op.a
                else:
                    input_op.a = input_op.a @ np.diag(a)
            elif isinstance(input_op.b, np.ndarray):
                if input_op.transpose_b:
                    input_op.b = np.diag(a) @ input_op.b
                else:
                    input_op.b = input_op.b @ np.diag(a)
            else:
                raise NotImplementedError()
            input_op.c = a * input_op.c + b
            return input_op
        elif isinstance(input_op, operations.Input):
            c = operation.mean.shape[0]
            std = np.sqrt(operation.variance + operation.epsilon)
            k = np.zeros((c, c, 1, 1))  # identity kernel (H, W, inC, outC)
            for i in range(c):
                k[i, i, 0, 0] = 1
            W = k * operation.scale / std
            b = operation.bias - operation.scale * operation.mean / std
            op = operations.Conv(input_op, W, b)
            return op
        # TODO : in what other scenarios can BatchNorm be converted?
        return operation
