from typing import List, Union

import numpy as np

from ... import OperationGraph, operations
from .base import Simplifier


class ReluifyMaxPool(Simplifier):
    @staticmethod
    def reluify_maxpool(
        operation: operations.MaxPool,
    ) -> Union[operations.Relu, operations.MaxPool]:
        if operation.ceil_mode:
            # TODO : can we support this?
            return operation
        if np.any(operation.dilations != 1):
            # TODO : can we support this?
            return operation
        input_op = operation.x
        kernel_shape = operation.kernel_shape
        k = int(np.product(kernel_shape))
        if k == 1:
            return operation

        (_, num_channels, *_), dtype = OperationGraph([input_op]).output_details[0]
        num_comparisons = int(np.ceil(k / 2))
        w = np.zeros((num_channels * num_comparisons * 3, num_channels, k), dtype=dtype)
        for channel_i in range(num_channels):
            for cmp_i in range(num_comparisons):
                index = np.ravel_multi_index(
                    (channel_i, cmp_i, 0), (num_channels, num_comparisons, 3)
                )
                idx1 = 2 * cmp_i
                idx2 = 2 * cmp_i + 1
                if idx2 < k:
                    w[index + 0, channel_i, idx1] = 1
                    w[index + 0, channel_i, idx2] = -1
                    w[index + 1, channel_i, idx2] = 1
                    w[index + 2, channel_i, idx2] = -1
                else:
                    w[index + 1, channel_i, idx1] = 1
                    w[index + 2, channel_i, idx1] = -1
        w = w.reshape(num_channels * num_comparisons * 3, num_channels, *kernel_shape)
        initial_conv = operations.Conv(
            input_op, w, strides=operation.strides, pads=operation.pads
        )
        initial_relu = operations.Relu(initial_conv)

        new_ops: List[Union[operations.Conv, operations.Relu]] = [initial_relu]
        while num_comparisons != 1:
            num_values = num_comparisons
            num_comparisons = int(np.ceil(num_values / 2))
            w = np.zeros(
                (
                    num_channels * num_comparisons * 3,
                    num_channels * num_values * 3,
                    1,
                    1,
                ),
                dtype=dtype,
            )
            for channel_i in range(num_channels):
                for cmp_i in range(num_comparisons):
                    index = np.ravel_multi_index(
                        (channel_i, cmp_i, 0), (num_channels, num_comparisons, 3)
                    )
                    idx1 = int(
                        np.ravel_multi_index(
                            (channel_i, 2 * cmp_i, 0), (num_channels, num_values, 3)
                        )
                    )
                    if 2 * cmp_i + 1 < num_values:
                        idx2 = int(
                            np.ravel_multi_index(
                                (channel_i, 2 * cmp_i + 1, 0),
                                (num_channels, num_values, 3),
                            )
                        )
                        w[index + 0, idx1 + 0, 0, 0] = 1
                        w[index + 0, idx1 + 1, 0, 0] = 1
                        w[index + 0, idx1 + 2, 0, 0] = -1

                        w[index + 0, idx2 + 0, 0, 0] = -1
                        w[index + 0, idx2 + 1, 0, 0] = -1
                        w[index + 0, idx2 + 2, 0, 0] = 1

                        w[index + 1, idx2 + 0, 0, 0] = 1
                        w[index + 1, idx2 + 1, 0, 0] = 1
                        w[index + 1, idx2 + 2, 0, 0] = -1

                        w[index + 2, idx2 + 0, 0, 0] = -1
                        w[index + 2, idx2 + 1, 0, 0] = -1
                        w[index + 2, idx2 + 2, 0, 0] = 1
                    else:
                        w[index + 1, idx1 + 0, 0, 0] = 1
                        w[index + 1, idx1 + 1, 0, 0] = 1
                        w[index + 1, idx1 + 2, 0, 0] = -1

                        w[index + 2, idx1 + 0, 0, 0] = -1
                        w[index + 2, idx1 + 1, 0, 0] = -1
                        w[index + 2, idx1 + 2, 0, 0] = 1
            new_ops.append(operations.Conv(new_ops[-1], w))
            new_ops.append(operations.Relu(new_ops[-1]))

        w = np.zeros((num_channels, num_channels * 3, 1, 1), dtype=dtype)
        for channel_i in range(num_channels):
            index = np.ravel_multi_index((channel_i, 0, 0), (num_channels, 1, 3))
            w[channel_i, index + 0, 0, 0] = 1
            w[channel_i, index + 1, 0, 0] = 1
            w[channel_i, index + 2, 0, 0] = -1
        new_ops.append(operations.Conv(new_ops[-1], w))

        new_op = operations.Relu(new_ops[-1])
        return new_op

    def visit_MaxPool(
        self, operation: operations.MaxPool
    ) -> Union[operations.Relu, operations.MaxPool]:
        if not isinstance(operation.x, operations.Relu):
            return operation
        return self.reluify_maxpool(operation)

    def visit_Relu(
        self, operation: operations.Relu
    ) -> Union[operations.Relu, operations.MaxPool]:
        if not isinstance(operation.x, operations.MaxPool):
            return operation
        return self.reluify_maxpool(operation.x)


__all__ = ["ReluifyMaxPool"]
