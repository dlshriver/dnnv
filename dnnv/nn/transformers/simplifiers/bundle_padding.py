import numpy as np

from copy import copy

from .base import Simplifier
from ... import operations


class BundlePadding(Simplifier):
    def visit_Conv(self, operation: operations.Conv) -> operations.Conv:
        input_op = operation.x
        if not isinstance(input_op, operations.Pad):
            return operation
        pads = operation.pads
        if input_op.mode != "constant" or input_op.value != 0.0:
            return operation
        if not np.all(p == 0 for p in input_op.pads[:4]):
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
        pads = operation.pads
        if input_op.mode != "constant" or input_op.value != 0.0:
            return operation
        if not np.all(p == 0 for p in input_op.pads[:4]):
            return operation
        operation = copy(operation)
        pad_top, pad_left = input_op.pads[2:4]
        pad_bottom, pad_right = input_op.pads[6:8]
        operation.pads = pads + np.array([pad_top, pad_left, pad_bottom, pad_right])
        operation.x = input_op.x
        return operation
