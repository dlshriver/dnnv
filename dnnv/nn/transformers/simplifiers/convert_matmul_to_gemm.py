from typing import Union

import numpy as np

from ... import operations
from ...graph import OperationGraph
from .base import Simplifier


class ConvertMatMulToGemm(Simplifier):
    def visit_MatMul(
        self, operation: operations.MatMul
    ) -> Union[operations.MatMul, operations.Gemm]:
        a = operation.a
        b = operation.b
        dtype = None
        if isinstance(a, operations.Operation):
            a_details = OperationGraph([a]).output_details[0]
            a_shape = a_details.shape
            dtype = a_details.dtype
        else:
            a_shape = np.shape(a)
        a_ndim = len(a_shape)
        if isinstance(b, operations.Operation):
            b_details = OperationGraph([b]).output_details[0]
            b_shape = b_details.shape
            dtype = b_details.dtype
        else:
            b_shape = np.shape(b)
        b_ndim = len(b_shape)
        if a_ndim == 2 and b_ndim == 2:
            c = np.zeros(b_shape[-1], dtype=dtype)
            return operations.Gemm(a, b, c)
        return operation


__all__ = ["ConvertMatMulToGemm"]
