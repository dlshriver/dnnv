from __future__ import annotations

import numpy as np

from collections import defaultdict
from typing import Callable, Dict, List, Optional, Set, Type, Union

from .base import OperationTransformer
from .. import operations
from ..graph import OperationGraph
from ..operations import Operation
from ..utils import TensorDetails


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


class FlatteningTransformer(OperationTransformer):
    def __init__(self, tensor_details):
        super().__init__(recursive=False)
        self._tensor_details: Optional[Dict[Operation, TensorDetails]] = tensor_details
        self._inputs = []

    def transform(self, op_graph: OperationGraph) -> OperationGraph:
        self._inputs = []
        new_op_graph = super().transform(op_graph)
        new_op_graph._inputs = self._inputs
        new_op_graph._input_details = [self._tensor_details[op] for op in self._inputs]
        new_op_graph._output_details = [
            self._tensor_details[op] for op in new_op_graph.outputs
        ]
        new_op_graph._is_linear = op_graph._is_linear
        return new_op_graph

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
        new_add_op = operations.Gemm(
            new_concat_op, np.block([[I], [I]]), np.zeros(size, dtype=dtype)
        )
        self._tensor_details[new_add_op] = self._tensor_details[operation]
        return self.visit(new_add_op)

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
        self._tensor_details[new_op] = self._tensor_details[operation]
        return new_op

    def visit_Input(self, operation: operations.Input) -> operations.Input:
        assert abs(operation.shape[0]) == 1
        new_op = operations.Input(
            (1, int(np.product(operation.shape[1:]))), operation.dtype
        )
        self._inputs.append(new_op)
        self._tensor_details[new_op] = TensorDetails(new_op.shape, new_op.dtype)
        return new_op


ActivationOperation = (
    operations.Elu,
    operations.LeakyRelu,
    operations.Relu,
    operations.Sigmoid,
    operations.Tanh,
)


class LinearizingTransformer(OperationTransformer):
    def __init__(self, tensor_details):
        super().__init__(recursive=False)
        self._tensor_details: Optional[Dict[Operation, TensorDetails]] = tensor_details

    def generic_visit(self, operation: Operation) -> Operation:
        result = super().generic_visit(operation)
        if result is not operation:
            self._tensor_details[result] = self._tensor_details[operation]
        return result

    def transform(self, op_graph: OperationGraph) -> OperationGraph:
        new_op_graph = super().transform(op_graph)
        new_op_graph._inputs = op_graph._inputs
        new_op_graph._input_details = op_graph._input_details
        new_op_graph._output_details = op_graph._output_details
        new_op_graph._is_linear = True
        return new_op_graph

    def _extract_computation_paths(self, operation: operations.Concat):
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
        if len([1 for k, v in uses.items() if len(v) > 1]) > 1:
            import warnings

            warnings.warn(
                "MLPTransformer does not efficiently optimize multiple computation paths with shared sub-paths"
            )
        base_op = max(uses.keys(), default=0, key=lambda k: len(uses[k]))
        computation_paths = [
            path[: op_to_index[base_op] + 1]
            for path, op_to_index in zip(computation_paths, op_to_path_index)
        ]
        return computation_paths

    def _path_parameters(self, computation_paths):
        first_layer_types = [
            type(path[-2]) for path in computation_paths[1:] if len(path) > 1
        ]
        if any(t != first_layer_types[0] for t in first_layer_types):
            raise NotImplementedError("TODO (A)!")
        if (
            len(first_layer_types) < len(computation_paths)
            and first_layer_types[0] != operations.Gemm
        ):
            raise NotImplementedError("TODO (E)!")
        longest_path_len = max(len(path) for path in computation_paths)
        path_parameters = []
        for path in computation_paths:
            parameters = []
            path_parameters.append(parameters)
            for op in reversed(path[:-1]):
                if isinstance(op, operations.Relu):
                    parameters.append(("relu",))
                elif isinstance(op, operations.Gemm):
                    parameters.append(("gemm", op.b, op.c))
                else:
                    raise NotImplementedError("TODO (C)!")
            if len(path) == longest_path_len:
                continue
            if (longest_path_len - len(parameters)) % 2 == 0:
                if len(parameters) == 0 or parameters[-1] == ("relu",):
                    details = self._tensor_details[path[0]]
                    size = details.shape[1]
                    dtype = details.dtype
                    parameters.append(
                        ("gemm", np.eye(size, dtype=dtype), np.zeros(size, dtype=dtype))
                    )
                elif parameters[-1][0] == "gemm":
                    parameters.append(("relu",))
                else:
                    raise NotImplementedError("TODO (C)!")
            while len(parameters) < longest_path_len - 1:
                if ("relu",) in parameters:
                    details = self._tensor_details[path[-1]]
                    size = details.shape[1]
                    dtype = details.dtype
                    if parameters[0] == ("relu",):
                        size *= 2
                    index = 0
                    for i, p in enumerate(parameters[:-1]):
                        if p == ("relu",):
                            continue
                        if p[1].shape[1] < size:
                            index = i + 1
                            size = p[1].shape[1]
                            dtype = p[1].dtype
                    b = np.eye(size, dtype=dtype)
                    c = np.zeros(size, dtype=dtype)
                    while len(parameters) < longest_path_len - 1:
                        parameters.insert(index + 1, ("relu",))
                        parameters.insert(index + 1, ("gemm", b, c))
                else:
                    assert len(parameters) == 1
                    details = self._tensor_details[path[-1]]
                    size = details.shape[1]
                    dtype = details.dtype
                    index = 0
                    if parameters[0][1].shape[1] < size:
                        index = -1
                        size = parameters[0][1].shape[1]
                        dtype = parameters[0][1].dtype
                    I = np.eye(size, dtype=dtype)
                    b = np.block([I, -I])
                    c = np.zeros(2 * size, dtype=dtype)
                    _, _b, _c = parameters.pop()
                    if index == 0:
                        parameters.append(("gemm", b, c))
                        parameters.append(("relu",))
                    else:
                        parameters.append(("gemm", _b @ b, _c @ b))
                        parameters.append(("relu",))
                    b = np.eye(2 * size, dtype=dtype)
                    while len(parameters) < longest_path_len - 3:
                        parameters.append(("gemm", b, c))
                        parameters.append(("relu",))
                    if index == 0:
                        b = np.block([[I], [-I]]) @ _b
                        parameters.append(("gemm", b, _c))
                    else:
                        b = np.block([[I], [-I]])
                        c = np.zeros(size, dtype=dtype)
                        parameters.append(("gemm", b, c))
                break
            if len(parameters) != longest_path_len - 1:
                raise NotImplementedError("TODO (B)!")
        return path_parameters

    def visit_Concat(self, operation: operations.Concat) -> operations.Operation:
        import itertools

        assert operation.axis == 1
        computation_paths = self._extract_computation_paths(operation)
        path_parameters = self._path_parameters(computation_paths)
        base_op = computation_paths[0][-1]
        num_paths = len(computation_paths)
        details = self._tensor_details[base_op]
        size = details.shape[1]
        dtype = details.dtype
        I = np.eye(size, dtype=dtype)
        base_op = operations.Gemm(
            base_op, np.block([I] * num_paths), np.zeros(size * num_paths, dtype=dtype)
        )
        from ..transformers.simplifiers.squeeze_gemms import SqueezeGemms

        base_op = SqueezeGemms().simplify(base_op)
        self._tensor_details[base_op] = TensorDetails(
            (details.shape[0], size * num_paths), dtype
        )
        for params in itertools.zip_longest(*path_parameters):
            op_type = params[0][0]
            if op_type == "relu":
                new_op = operations.Relu(base_op)
                self._tensor_details[new_op] = self._tensor_details[base_op]
            elif op_type == "gemm":
                b = diagonalize(*(p[1] for p in params))
                c = np.block([p[2] for p in params])
                new_op = operations.Gemm(base_op, b, c)
                new_op = SqueezeGemms().simplify(new_op)
                base_details = self._tensor_details[base_op]
                shape = base_details.shape[0], b.shape[1]
                dtype = base_details.dtype
                self._tensor_details[new_op] = TensorDetails(shape, dtype)
            base_op = new_op
        return base_op


class MLPTransformer(OperationTransformer):
    def __init__(self):
        super().__init__(recursive=False)
        self._tensor_details: Optional[Dict[Operation, TensorDetails]] = None

    def transform(self, op_graph: OperationGraph) -> OperationGraph:
        simplified_op_graph = op_graph.simplify()
        _ = simplified_op_graph.tensor_details
        flattened_op_graph = FlatteningTransformer(
            simplified_op_graph.tensor_details
        ).transform(simplified_op_graph)
        if len(flattened_op_graph.outputs) > 1:
            op = op_graph.outputs[0]
            shape = flattened_op_graph.tensor_details[op].shape
            dtype = flattened_op_graph.tensor_details[op].dtype
            for op in op_graph.outputs[1:]:
                shape[1] += flattened_op_graph.tensor_details[op].shape[1]
            output_op = operations.Concat(op_graph.outputs, axis=1)
            op_graph = OperationGraph([output_op])
            op_graph._inputs = flattened_op_graph._inputs
            op_graph._input_details = flattened_op_graph._input_details
            op_graph._output_details = [TensorDetails(shape, dtype)]
            op_graph._is_linear = flattened_op_graph._is_linear
            op_graph._tensor_details = flattened_op_graph._tensor_details
            flattened_op_graph = op_graph
        _ = flattened_op_graph.tensor_details
        linearized_op_graph = LinearizingTransformer(
            flattened_op_graph.tensor_details
        ).transform(flattened_op_graph)
        mlp_op_graph = linearized_op_graph.simplify()
        return mlp_op_graph


__all__ = ["MLPTransformer"]
