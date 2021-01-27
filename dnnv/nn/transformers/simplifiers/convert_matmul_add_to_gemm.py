from typing import Union

from .base import Simplifier
from ... import operations
from ...graph import OperationGraph


class ConvertMatMulAddToGemm(Simplifier):
    def visit_Add(
        self, operation: operations.Add
    ) -> Union[operations.Add, operations.Gemm]:
        if isinstance(operation, operations.Add):
            if isinstance(operation.a, operations.Operation):
                input_op = operation.a
                c = operation.b
            else:
                input_op = operation.b
                c = operation.a
            if isinstance(input_op, operations.MatMul):
                a = input_op.a
                b = input_op.b
                if isinstance(a, operations.Operation):
                    a_ndim = len(OperationGraph([a]).output_shape[0])
                else:
                    a_ndim = len(a.shape)
                if isinstance(b, operations.Operation):
                    b_ndim = len(OperationGraph([b]).output_shape[0])
                else:
                    b_ndim = len(b.shape)
                if a_ndim == 2 and b_ndim == 2:
                    return operations.Gemm(a, b, c)
        return operation

