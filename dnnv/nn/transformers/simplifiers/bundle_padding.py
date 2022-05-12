from copy import copy

import numpy as np

from ... import operations
from .base import Simplifier


class BundlePadding(Simplifier):
    def visit_Conv(self, operation: operations.Conv) -> operations.Conv:
        input_op = operation.x
        if not isinstance(input_op, operations.Pad):
            return operation
        if all(p == 0 for p in input_op.pads):
            operation = copy(operation)
            operation.x = input_op.x
            return operation
        pads = operation.pads
        if input_op.mode != "constant" or input_op.value != 0.0:
            return operation
        num_pads = len(input_op.pads)
        if any(p != 0 for p in input_op.pads[:2]) or any(
            p != 0 for p in input_op.pads[num_pads // 2 : num_pads // 2 + 2]
        ):
            return operation
        operation = copy(operation)
        pad_top, pad_left = input_op.pads[2:4]
        pad_bottom, pad_right = input_op.pads[6:8]
        operation.pads = pads + np.array([pad_top, pad_left, pad_bottom, pad_right])
        operation.x = input_op.x
        return operation

    def visit_MaxPool(self, operation: operations.MaxPool) -> operations.MaxPool:
        input_op = operation.x
        if not isinstance(input_op, operations.Pad):
            return operation
        if all(p == 0 for p in input_op.pads):
            operation = copy(operation)
            operation.x = input_op.x
            return operation
        num_pads = len(input_op.pads)
        if any(p != 0 for p in input_op.pads[:2]) or any(
            p != 0 for p in input_op.pads[num_pads // 2 : num_pads // 2 + 2]
        ):
            return operation
        if input_op.mode != "constant" or not np.isnan(input_op.value):
            return operation
        pads = operation.pads
        operation = copy(operation)
        pad_top, pad_left = input_op.pads[2:4]
        pad_bottom, pad_right = input_op.pads[6:8]
        operation.pads = pads + np.array([pad_top, pad_left, pad_bottom, pad_right])
        operation.x = input_op.x
        return operation


__all__ = ["BundlePadding"]
