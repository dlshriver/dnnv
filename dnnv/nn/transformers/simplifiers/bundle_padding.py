import numpy as np

from ... import operations
from .base import Simplifier


class BundlePadding(Simplifier):
    def visit_Conv(self, operation: operations.Conv) -> operations.Conv:
        input_op = operation.x
        if not isinstance(input_op, operations.Pad):
            return operation
        if all(p == 0 for p in input_op.pads):
            operation = operations.Conv(
                input_op.x,
                operation.w,
                operation.b,
                dilations=operation.dilations,
                group=operation.group,
                kernel_shape=operation.kernel_shape,
                pads=operation.pads,
                strides=operation.strides,
            )
            return operation
        pads = operation.pads
        if input_op.mode != "constant" or input_op.value != 0.0:
            return operation
        num_pads = len(input_op.pads)
        if any(p != 0 for p in input_op.pads[:2]) or any(
            p != 0 for p in input_op.pads[num_pads // 2 : num_pads // 2 + 2]
        ):
            return operation
        pad_top, pad_left = input_op.pads[2:4]
        pad_bottom, pad_right = input_op.pads[6:8]
        new_pads = pads + np.array([pad_top, pad_left, pad_bottom, pad_right])
        operation = operations.Conv(
            input_op.x,
            operation.w,
            operation.b,
            dilations=operation.dilations,
            group=operation.group,
            kernel_shape=operation.kernel_shape,
            pads=new_pads,
            strides=operation.strides,
        )
        return operation

    def visit_MaxPool(self, operation: operations.MaxPool) -> operations.MaxPool:
        input_op = operation.x
        if not isinstance(input_op, operations.Pad):
            return operation
        if all(p == 0 for p in input_op.pads):
            operation = operations.MaxPool(
                input_op.x,
                kernel_shape=operation.kernel_shape,
                ceil_mode=operation.ceil_mode,
                dilations=operation.dilations,
                pads=operation.pads,
                storage_order=operation.storage_order,
                strides=operation.strides,
                name=operation.name,
            )
            return operation
        num_pads = len(input_op.pads)
        if any(p != 0 for p in input_op.pads[:2]) or any(
            p != 0 for p in input_op.pads[num_pads // 2 : num_pads // 2 + 2]
        ):
            return operation
        if input_op.mode != "constant" or (
            not np.isnan(input_op.value) and not np.isneginf(input_op.value)
        ):
            return operation
        pads = operation.pads
        pad_top, pad_left = input_op.pads[2:4]
        pad_bottom, pad_right = input_op.pads[6:8]
        new_pads = pads + np.array([pad_top, pad_left, pad_bottom, pad_right])
        operation = operations.MaxPool(
            input_op.x,
            kernel_shape=operation.kernel_shape,
            ceil_mode=operation.ceil_mode,
            dilations=operation.dilations,
            pads=new_pads,
            storage_order=operation.storage_order,
            strides=operation.strides,
            name=operation.name,
        )
        return operation


__all__ = ["BundlePadding"]
