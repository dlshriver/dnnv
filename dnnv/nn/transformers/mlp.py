from __future__ import annotations

import numpy as np

from collections import defaultdict
from scipy.linalg import block_diag
from typing import Dict, List, Optional, Sequence, Type


from .base import OperationTransformer
from .. import operations
from ..graph import OperationGraph
from ..operations import Operation
from ..transformers.simplifiers.squeeze_gemms import SqueezeGemms
from ..utils import TensorDetails


ActivationOperation = (
    operations.Elu,
    operations.LeakyRelu,
    operations.Relu,
    operations.Sigmoid,
    operations.Tanh,
)

GeneratorOperation = (operations.Input, operations.RandomUniform)


class MLPTransformer(OperationTransformer):
    def __init__(self):
        super().__init__(recursive=False)
        self._inputs: Dict[operations.Input, int] = {}
        self._tensor_details: Optional[Dict[Operation, TensorDetails]] = None

    def generic_visit(self, operation: Operation) -> Operation:
        result = super().generic_visit(operation)
        if result is not operation:
            self._tensor_details[result] = self._tensor_details[operation]
        return result

    def visit_Add(self, operation: operations.Add):
        if not isinstance(operation.a, operations.Operation) or not isinstance(
            operation.b, operations.Operation
        ):
            return super().visit(operation)
        a_shape, a_dtype = self._tensor_details[operation.a]
        b_shape, b_dtype = self._tensor_details[operation.b]
        assert a_shape == b_shape
        assert a_shape[0] == 1
        dtype = np.result_type(a_dtype, b_dtype)
        size = np.product(a_shape)
        I = np.eye(size, dtype=dtype)
        a = operation.a
        if len(a_shape) > 2:
            a = operations.Flatten(operation.a)
            a_shape = (a_shape[0], int(np.product(a_shape[1:])))
            self._tensor_details[a] = TensorDetails(a_shape, a_dtype)
        b = operation.b
        if len(b_shape) > 2:
            b = operations.Flatten(operation.b)
            b_shape = (b_shape[0], int(np.product(b_shape[1:])))
            self._tensor_details[b] = TensorDetails(b_shape, b_dtype)
        new_concat_op = operations.Concat([a, b], axis=1)
        self._tensor_details[new_concat_op] = TensorDetails(
            (a_shape[0], a_shape[1] + b_shape[1]), a_dtype
        )
        new_concat_op = self.visit(new_concat_op)

        new_add_op = operations.Gemm(
            new_concat_op, np.block([[I], [I]]), np.zeros(size, dtype=dtype)
        )
        self._tensor_details[new_add_op] = self._tensor_details[operation]

        new_add_op = SqueezeGemms().simplify(new_add_op)
        self._tensor_details[new_add_op] = self._tensor_details[operation]
        return new_add_op

    def visit_Concat(self, operation: operations.Concat) -> operations.Operation:
        assert operation.axis == 1
        computation_paths = self._extract_computation_paths(operation)
        if len(set(path[0] for path in computation_paths)) == 1:
            base_op = computation_paths[0][0]
            details = self._tensor_details[base_op]
            size = details.shape[1]
            dtype = details.dtype
            I = np.eye(size, dtype=dtype)
            z = np.zeros(size, dtype=dtype)
            b = np.block([I for _ in computation_paths])
            c = np.hstack([z for _ in computation_paths])
            gemm_op = operations.Gemm(base_op, b, c)
            new_op = SqueezeGemms().simplify(gemm_op)
        elif all(isinstance(path[0], GeneratorOperation) for path in computation_paths):
            new_op = operation
        elif all(isinstance(path[0], operations.Relu) for path in computation_paths):
            concat_op = operations.Concat(
                [path[1] for path in computation_paths], axis=1
            )
            self._tensor_details[concat_op] = self._tensor_details[operation]
            new_op = operations.Relu(self.visit(concat_op))
        elif all(isinstance(path[0], operations.Gemm) for path in computation_paths):
            b = block_diag(*(path[0].b for path in computation_paths))
            c = np.hstack([path[0].c for path in computation_paths])
            concat_op = operations.Concat(
                [path[1] for path in computation_paths], axis=1
            )
            self._tensor_details[concat_op] = self._tensor_details[operation]
            gemm_op = operations.Gemm(self.visit(concat_op), b, c)
            new_op = SqueezeGemms().simplify(gemm_op)
        elif any(isinstance(path[0], operations.Gemm) for path in computation_paths):
            B = []
            C = []
            input_ops = []
            for path in computation_paths:
                op = path[0]
                if isinstance(op, operations.Gemm):
                    B.append(op.b)
                    C.append(op.c)
                    input_ops.append(path[1])
                else:
                    details = self._tensor_details[op]
                    size = details.shape[1]
                    dtype = details.dtype
                    B.append(np.eye(size, dtype=dtype))
                    C.append(np.zeros(size, dtype=dtype))
                    input_ops.append(op)
            b = block_diag(*B)
            c = np.hstack(C)
            concat_op = operations.Concat(input_ops, axis=1)
            self._tensor_details[concat_op] = self._tensor_details[operation]
            gemm_op = operations.Gemm(self.visit(concat_op), b, c)
            new_op = SqueezeGemms().simplify(gemm_op)
        elif any(isinstance(path[0], operations.Relu) for path in computation_paths):
            input_ops = []
            for path in computation_paths:
                op = path[0]
                if isinstance(op, operations.Relu):
                    details = self._tensor_details[op]
                    size = details.shape[1]
                    dtype = details.dtype
                    b = np.eye(size, dtype=dtype)
                    c = np.zeros(size, dtype=dtype)
                    gemm_op = operations.Gemm(op, b, c)
                    gemm_op = SqueezeGemms().simplify(gemm_op)
                    self._tensor_details[gemm_op] = self._tensor_details[op]
                    input_ops.append(gemm_op)
                elif isinstance(op, operations.RandomUniform):
                    details = self._tensor_details[op]
                    size = details.shape[1]
                    dtype = details.dtype
                    lb = op.low
                    ub = op.high
                    b = np.eye(size, dtype=dtype) * (ub - lb)
                    c = np.full(size, lb, dtype=dtype)
                    relu_op = operations.Relu(op)
                    gemm_op = operations.Gemm(relu_op, b, c)
                    gemm_op = SqueezeGemms().simplify(gemm_op)
                    self._tensor_details[relu_op] = self._tensor_details[op]
                    self._tensor_details[gemm_op] = self._tensor_details[op]
                    input_ops.append(gemm_op)
                else:
                    details = self._tensor_details[op]
                    size = details.shape[1]
                    dtype = details.dtype
                    I = np.eye(size, dtype=dtype)
                    z = np.zeros(size, dtype=dtype)
                    z2 = np.zeros(size * 2, dtype=dtype)
                    gemm_op_1 = operations.Gemm(op, np.block([I, -I]), z2)
                    gemm_op_1 = SqueezeGemms().simplify(gemm_op_1)
                    relu_op = operations.Relu(gemm_op_1)
                    gemm_op_2 = operations.Gemm(
                        relu_op,
                        np.block([[I], [-I]]),
                        z,
                    )
                    gemm_op_2 = SqueezeGemms().simplify(gemm_op_2)
                    self._tensor_details[gemm_op_1] = TensorDetails(
                        (details.shape[0], size * 2), dtype
                    )
                    self._tensor_details[relu_op] = self._tensor_details[gemm_op_1]
                    self._tensor_details[gemm_op_2] = self._tensor_details[op]
                    input_ops.append(gemm_op_2)
            concat_op = operations.Concat(input_ops, axis=1)
            self._tensor_details[concat_op] = self._tensor_details[operation]
            new_op = self.visit(concat_op)
        else:
            assert 0, f"TODO B: {[path[0] for path in computation_paths]}"
        self._tensor_details[new_op] = self._tensor_details[operation]
        return new_op

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

        _, in_c, in_h, in_w = self._tensor_details[operation.x].shape
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
        new_op = operations.Gemm(x, gemm_b, gemm_c)
        new_op = SqueezeGemms().simplify(new_op)
        old_details = self._tensor_details[operation]
        self._tensor_details[new_op] = TensorDetails(
            (old_details.shape[0], output_size), old_details.dtype
        )
        return new_op

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
        new_op = operations.Gemm(a, b, c)
        new_op = SqueezeGemms().simplify(new_op)
        self._tensor_details[new_op] = self._tensor_details[operation]
        return new_op

    def visit_Input(self, operation: operations.Input) -> operations.Input:
        assert abs(operation.shape[0]) == 1
        if operation in self._inputs:
            return operation
        new_op = operations.Input(
            (1, int(np.product(operation.shape[1:]))), operation.dtype
        )
        self._inputs[new_op] = len(self._inputs)
        self._tensor_details[new_op] = TensorDetails(new_op.shape, new_op.dtype)
        return new_op

    def visit_RandomUniform(
        self, operation: operations.RandomUniform
    ) -> operations.RandomUniform:
        assert len(self._inputs) > 0
        assert abs(operation.shape[0]) == 1
        low = operation.low
        high = operation.high
        if low >= 0:
            new_op = operations.RandomUniform(
                (1, int(np.product(operation.shape[1:]))),
                dtype=operation.dtype,
                low=low,
                high=high,
            )
            self._tensor_details[new_op] = TensorDetails(new_op.shape, new_op.dtype)
        else:
            size = int(np.product(operation.shape[1:]))
            rand_op = operations.RandomUniform((1, size), dtype=operation.dtype)
            w = np.diagonal(np.full(size, high - low))
            b = np.full(size, low)
            new_op = operations.Gemm(rand_op, w, b)
            self._tensor_details[rand_op] = TensorDetails(rand_op.shape, rand_op.dtype)
            self._tensor_details[new_op] = TensorDetails(rand_op.shape, rand_op.dtype)
        return new_op

    def transform(
        self, op_graph: OperationGraph, *, _pre_simplified=False
    ) -> OperationGraph:
        assert len(op_graph.outputs) == 1
        import time

        start_t = time.time()
        self._inputs = {}
        simplified_op_graph = op_graph
        if not _pre_simplified:
            simplified_op_graph = op_graph.simplify()
        self._tensor_details = simplified_op_graph.tensor_details
        mlp_op_graph = super().transform(simplified_op_graph)
        mlp_op_graph._inputs = tuple(self._inputs)
        mlp_op_graph._input_details = tuple(
            self._tensor_details[op] for op in self._inputs
        )
        mlp_op_graph._output_details = tuple(
            self._tensor_details[op] for op in mlp_op_graph.outputs
        )
        mlp_op_graph._tensor_details = self._tensor_details
        mlp_op_graph._is_linear = True
        mlp_op_graph._simplified = mlp_op_graph
        return mlp_op_graph

    def _extract_computation_paths(
        self, operation: operations.Concat
    ) -> Sequence[Sequence[Operation]]:
        activation: Optional[Type[Operation]] = None
        num_inputs = len(operation.x)
        computation_paths: List[List[Operation]] = [
            [self.visit(op)] for op in operation.x
        ]
        op_to_path_index: List[Dict[Optional[Operation], int]] = [
            {op: 0} for [op] in computation_paths
        ]
        uses: Dict[Optional[Operation], List[Operation]] = defaultdict(list)
        for [op] in computation_paths:
            uses[op].append(operation)
        while len(max(uses.values(), default=0, key=len)) < num_inputs:
            for path_id, path in enumerate(computation_paths):
                op = path[-1]
                if op is None:
                    continue
                if isinstance(op, ActivationOperation):
                    if activation is None:
                        activation = type(op)
                        assert (
                            activation == operations.Relu
                        ), "only Relu MLPs are currently supported"
                    assert type(op) == activation
                assert len(op.inputs) <= 1
                if len(op.inputs) == 0:
                    input_op = None
                else:
                    input_op = op.inputs[0]
                uses[input_op].append(op)
                path.append(input_op)
                assert input_op not in op_to_path_index[path_id]
                op_to_path_index[path_id][input_op] = len(op_to_path_index[path_id])
                if len(uses[input_op]) == num_inputs:
                    break
        base_op = max(uses.keys(), default=0, key=lambda k: len(uses[k]))
        computation_paths = [
            path[: op_to_index[base_op] + 1]
            for path, op_to_index in zip(computation_paths, op_to_path_index)
        ]
        return computation_paths


__all__ = ["MLPTransformer"]
