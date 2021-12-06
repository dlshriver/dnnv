import numpy as np

from typing import Union

from .base import Simplifier
from ... import operations
from ...graph import OperationGraph


class ConvertMatMulToGemm(Simplifier):
    def visit_MatMul(
        self, operation: operations.MatMul
    ) -> Union[operations.MatMul, operations.Gemm]:
        a = operation.a
        b = operation.b
        if isinstance(a, operations.Operation):
            a_details = OperationGraph([a]).output_details[0]
            a_shape = a_details.shape
        else:
            a_shape = np.shape(a)
        a_ndim = len(a_shape)
        if isinstance(b, operations.Operation):
            b_shape = OperationGraph([b]).output_shape[0]
        else:
            b_shape = np.shape(b)
        b_ndim = len(b_shape)
        if a_ndim == 2 and b_ndim == 2:
            return operations.Gemm(a, b)
        return operation


__all__ = ["ConvertMatMulToGemm"]
