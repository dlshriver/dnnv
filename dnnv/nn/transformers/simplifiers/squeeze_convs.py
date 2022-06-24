from copy import copy

import numpy as np

from ... import operations
from .base import Simplifier


class SqueezeConvs(Simplifier):
    def is_diagonal(self, array):
        i, j = array.shape
        return (i == j) and ~np.any(array.reshape(-1)[:-1].reshape(i - 1, j + 1)[:, 1:])

    def visit_Conv(self, operation: operations.Conv) -> operations.Conv:
        if (
            isinstance(operation.x, operations.Conv)
            and operation.x.w.shape[2] == 1
            and operation.x.w.shape[3] == 1
            and all(p == 0 for p in operation.pads)
            and all(s == 1 for s in operation.x.strides)
            and all(p == 0 for p in operation.x.pads)
            and all(d == 1 for d in operation.x.dilations)
            and operation.x.group == 1
            and self.is_diagonal(operation.x.w[:, :, 0, 0])
        ):
            w = np.diag(operation.x.w[:, :, 0, 0]).reshape((1, -1, 1, 1))
            b = operation.x.b

            out_c, in_c, k_h, k_w = operation.w.shape

            weights = operation.w * np.tile(w, (out_c, 1, k_h, k_w))
            bias = operation.b + (
                operation.w * np.tile(b.reshape((1, -1, 1, 1)), (out_c, 1, k_h, k_w))
            ).sum(axis=(1, 2, 3))

            op = copy(operation)
            op.x = operation.x.x
            op.w = weights
            op.b = bias

            return op
        return operation


__all__ = ["SqueezeConvs"]
