from __future__ import annotations

import numpy as np

from typing import Dict, List, Set, Union

from .base import OperationTransformer
from .. import operations
from ..graph import OperationGraph


def diagonalize(*A):
    blocks = [[None] * len(A) for _ in range(len(A))]
    for i, a1 in enumerate(A):
        for j, a2 in enumerate(A[i:], i):
            if i == j:
                blocks[i][j] = a1
            else:
                blocks[i][j] = np.zeros((a1.shape[0], a2.shape[1]), dtype=a1.dtype)
                blocks[j][i] = np.zeros((a2.shape[0], a1.shape[1]), dtype=a1.dtype)
    return np.block(blocks)


class MLPTransformer(OperationTransformer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.uses: Dict[operations.Operation, Set[operations.Operation]] = {}

    def transform(self, op_graph: OperationGraph) -> OperationGraph:
        simplified_op_graph = op_graph.simplify()
        mlp_op_graph = super().transform(simplified_op_graph).simplify()
        return mlp_op_graph

    def visit_Concat(
        self, operation: operations.Concat
    ) -> Union[operations.Gemm, operations.Relu]:
        assert operation.axis == 1
        x = [self.visit(op) for op in operation.x]
        if len(set(x)) == 1:
            a = x[0]
            if isinstance(a, operations.Input):
                n = a.shape[-1]
                dtype = a.dtype
            else:
                details = OperationGraph([a]).output_details[0]
                n = details.shape[-1]
                dtype = details.dtype
            return operations.Gemm(
                a,
                np.block([np.eye(n, dtype=dtype) for _ in x]),
                np.zeros(n * len(x), dtype=dtype),
            )
        if all(isinstance(op, operations.Relu) for op in x):
            return operations.Relu(
                self.visit(operations.Concat([op.x for op in x], axis=operation.axis))
            )
        if all(isinstance(op, operations.Gemm) for op in x):
            return operations.Gemm(
                self.visit(operations.Concat([op.a for op in x], axis=operation.axis)),
                diagonalize(*(op.b for op in x)),
                np.block([op.c for op in x]),
            )
        if any(isinstance(op, operations.Relu) for op in x):
            inputs: List[operations.Operation] = []
            blocks = []
            for op in x:
                if isinstance(op, operations.Input):
                    n = op.shape[-1]
                    dtype = op.dtype
                else:
                    details = OperationGraph([op]).output_details[0]
                    n = details.shape[-1]
                    dtype = details.dtype
                I = np.eye(n, dtype=dtype)
                if isinstance(op, operations.Relu):
                    inputs.append(op)
                    blocks.append(I)
                else:
                    inputs.append(
                        operations.Relu(
                            operations.Gemm(
                                op, np.block([I, -I]), np.zeros(2 * n, dtype=dtype)
                            )
                        )
                    )
                    blocks.append(np.block([[I], [-I]]))
            W = diagonalize(*blocks)
            return operations.Gemm(
                self.visit(operations.Concat(inputs, axis=operation.axis)),
                W,
                np.zeros(W.shape[1], dtype=W.dtype),
            )
        inputs = []
        for op in x:
            if isinstance(op, operations.Gemm):
                inputs.append(op)
            else:
                if isinstance(op, operations.Input):
                    n = op.shape[-1]
                    dtype = op.dtype
                else:
                    details = OperationGraph([op]).output_details[0]
                    n = details.shape[-1]
                    dtype = details.dtype
                inputs.append(
                    operations.Gemm(
                        op, np.eye(n, dtype=dtype), np.zeros(n, dtype=dtype)
                    )
                )
        return self.visit(operations.Concat(inputs, axis=operation.axis))

    def visit_Gemm(self, operation: operations.Gemm) -> operations.Gemm:
        a = operation.a
        assert isinstance(a, operations.Operation)
        assert not operation.transpose_a
        a = self.visit(a)
        b = operation.b
        assert isinstance(b, np.ndarray)
        if operation.transpose_b:
            b = np.transpose(b)
        c = operation.c
        assert isinstance(c, np.ndarray)
        assert operation.alpha == 1
        assert operation.beta == 1
        return operations.Gemm(a, b, c)


__all__ = ["MLPTransformer"]
