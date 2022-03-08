from __future__ import annotations

import numpy as np

from typing import Callable, Dict, List, Set, Union

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
        if len(op_graph.outputs) > 1:
            output_operations = [operations.Flatten(o) for o in op_graph.outputs]
            output_op = operations.Concat(output_operations, axis=1)
            op_graph = OperationGraph([output_op])
        simplified_op_graph = op_graph.simplify()
        mlp_op_graph = super().transform(simplified_op_graph)
        simplified_mlp_op_graph = mlp_op_graph.simplify()
        return simplified_mlp_op_graph

    def visit_Add(self, operation: operations.Add):
        if not isinstance(operation.a, operations.Operation) or not isinstance(
            operation.b, operations.Operation
        ):
            return super().visit(operation)
        a_shape, a_dtype = OperationGraph([operation.a]).output_details[0]
        b_shape, b_dtype = OperationGraph([operation.b]).output_details[0]
        assert a_shape == b_shape
        assert a_shape[0] == 1
        dtype = np.result_type(a_dtype, b_dtype)
        size = np.product(a_shape)
        I = np.eye(size, dtype=dtype)
        new_concat_op = operations.Concat(
            [operations.Flatten(operation.a), operations.Flatten(operation.b)], axis=1
        )
        new_add_op = operations.Gemm(
            new_concat_op, np.block([[I], [I]]), np.zeros(size, dtype=dtype)
        )
        return self.visit(new_add_op)

    def visit_Concat(self, operation: operations.Concat) -> operations.Operation:
        assert operation.axis == 1
        new_op_constructor: Callable = lambda x: x
        x = [self.visit(op) for op in operation.x]
        while len(set(x)) > 1:
            if all(isinstance(op, operations.Input) for op in x):
                assert all(len(op.shape) == 2 for op in x)
                assert all(op.shape[0] == 1 for op in x)
                assert all(op.dtype == x[0].dtype for op in x)
                input_ops: Dict[operations.Input, int] = {}
                blocks: List[List[np.ndarray]] = []
                for op in x:
                    n = op.shape[1]
                    if op not in input_ops:
                        input_ops[op] = len(input_ops)
                        r = (
                            [
                                np.zeros((n, block.shape[1]), dtype=op.dtype)
                                for block in blocks[0]
                            ]
                            if len(blocks)
                            else []
                        )
                        r.append(np.eye(n, dtype=op.dtype))
                        for block in blocks:
                            block.append(
                                np.zeros((block[0].shape[0], n), dtype=op.dtype)
                            )
                        blocks.append(r)
                    else:
                        for i, block in enumerate(blocks):
                            if i == input_ops[op]:
                                block.append(np.eye(n, dtype=op.dtype))
                            else:
                                block.append(
                                    np.zeros((block[0].shape[0], n), dtype=op.dtype)
                                )
                b = np.block(blocks)
                c = np.zeros(b.shape[1], dtype=b.dtype)
                new_op_constructor = lambda x, op_constructor=new_op_constructor, b=b, c=c: op_constructor(
                    operations.Gemm(x, b, c)
                )
                x = [
                    operations.Input(
                        (1, sum(op.shape[1] for op in input_ops)), x[0].dtype
                    )
                ]
            elif any(
                all(isinstance(op, act_fn) for op in x)
                for act_fn in (
                    operations.Elu,
                    operations.Relu,
                    operations.Sigmoid,
                    operations.Tanh,
                )
            ):
                op_type = type(x[0])
                new_op_constructor = (
                    lambda x, op_constructor=new_op_constructor: op_constructor(
                        op_type(x)
                    )
                )
                x = [op.x for op in x]
            elif all(isinstance(op, operations.Gemm) for op in x):
                b = diagonalize(*(op.b for op in x))
                c = np.block([op.c for op in x])
                new_op_constructor = lambda x, op_constructor=new_op_constructor, b=b, c=c: op_constructor(
                    operations.Gemm(x, b, c)
                )
                x = [op.a for op in x]
            elif any(isinstance(op, operations.Relu) for op in x):
                inputs: List[operations.Operation] = []
                weight_blocks = []
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
                        weight_blocks.append(I)
                    else:
                        inputs.append(
                            operations.Relu(
                                operations.Gemm(
                                    op, np.block([I, -I]), np.zeros(2 * n, dtype=dtype)
                                )
                            )
                        )
                        weight_blocks.append(np.block([[I], [-I]]))
                b = diagonalize(*weight_blocks)
                c = np.zeros(b.shape[1], dtype=b.dtype)
                new_op_constructor = lambda x, op_constructor=new_op_constructor, b=b, c=c: op_constructor(
                    operations.Gemm(x, b, c)
                )
                x = inputs
            elif any(isinstance(op, operations.Input) for op in x):
                x = [
                    op
                    if not isinstance(op, operations.Input)
                    else operations.Gemm(
                        op,
                        np.eye(op.shape[1], dtype=op.dtype),
                        np.zeros(op.shape[1], dtype=op.dtype),
                    )
                    for op in x
                ]
            else:
                print("\n\nERROR!!!")
                OperationGraph([operations.Concat(x, 1)]).pprint()
                raise RuntimeError(f"{x}")
        a = x[0]
        if isinstance(a, operations.Input):
            n = a.shape[-1]
            dtype = a.dtype
        else:
            details = OperationGraph([a]).output_details[0]
            n = details.shape[-1]
            dtype = details.dtype
        I = np.eye(n, dtype=dtype)
        return new_op_constructor(
            operations.Gemm(
                a,
                np.block([I for _ in x]),
                np.zeros(n * len(x), dtype=dtype),
            )
        )

    def visit_Conv(self, operation: operations.Conv) -> operations.Gemm:
        x = operation.x
        assert isinstance(x, operations.Operation)
        x = self.visit(x)
        w = operation.w
        assert isinstance(w, np.ndarray)
        b = operation.b
        assert isinstance(b, np.ndarray)
        assert np.all(operation.group == 1)
        assert np.all(operation.dilations == 1)

        k_h, k_w = w.shape[2:4]
        p_top, p_left, p_bottom, p_right = operation.pads
        s_h, s_w = operation.strides

        _, in_c, in_h, in_w = OperationGraph([operation.x]).output_shape[0]
        input_size = in_c * in_h * in_w
        out_c = w.shape[0]
        out_h = (in_h - k_h + p_top + p_bottom) // s_h + 1
        out_w = (in_w - k_w + p_left + p_right) // s_w + 1
        output_size = out_c * out_h * out_w

        gemm_b = np.zeros((input_size, output_size), dtype=w.dtype)
        gemm_c = np.tile(b[..., None, None], (1, out_h, out_w)).flatten()
        for co in range(out_c):
            for ho in range(out_h):
                for wo in range(out_w):
                    out_idx = np.ravel_multi_index((co, ho, wo), (out_c, out_h, out_w))
                    for hk in range(k_h):
                        for wk in range(k_w):
                            for ck in range(in_c):
                                hi = ho * s_h + hk - p_top
                                wi = wo * s_w + wk - p_left
                                if hi < 0 or hi >= in_h:
                                    continue
                                if wi < 0 or wi >= in_w:
                                    continue
                                in_idx = np.ravel_multi_index(
                                    (ck, hi, wi), (in_c, in_h, in_w)
                                )
                                gemm_b[in_idx, out_idx] = w[co, ck, hk, wk]
        return operations.Gemm(x, gemm_b, gemm_c)

    def visit_Flatten(self, operation: operations.Flatten) -> operations.Operation:
        return self.visit(operation.x)

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

    def visit_Input(self, operation: operations.Input) -> operations.Input:
        assert abs(operation.shape[0]) == 1
        return operations.Input(
            (1, int(np.product(operation.shape[1:]))), operation.dtype
        )


__all__ = ["MLPTransformer"]
