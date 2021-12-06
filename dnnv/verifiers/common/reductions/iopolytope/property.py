from __future__ import annotations

import numpy as np

from typing import List, Optional, Tuple, Union

from .base import HalfspacePolytope, HyperRectangle
from ..base import Property
from .....nn import OperationGraph, OperationTransformer, operations
from .....properties import Network


class IOPolytopeProperty(Property):
    def __init__(
        self,
        networks: List[Network],
        input_constraint: HalfspacePolytope,
        output_constraint: HalfspacePolytope,
    ):
        self.networks = networks
        self.input_constraint = input_constraint
        setattr(self.input_constraint, "_varname", "x")
        self.output_constraint = output_constraint
        setattr(
            self.output_constraint,
            "_varname",
            [f"{network}(x)" for network in self.networks],
        )
        # TODO : move Merger out of this function
        class Merger(OperationTransformer):
            # TODO : merge common layers (e.g. same normalization, reshaping of input)
            def __init__(self):
                super().__init__()
                self.output_operations = []
                self.input_operations = {}

            def merge(self, operation_graphs: List[OperationGraph]):
                for op_graph in operation_graphs:
                    for op in op_graph.output_operations:
                        self.output_operations.append(self.visit(op))
                return OperationGraph(self.output_operations)

            def visit_Input(self, operation):
                input_details = (operation.dtype, tuple(operation.shape))
                if input_details not in self.input_operations:
                    self.input_operations[input_details] = self.generic_visit(operation)
                return self.input_operations[input_details]

        self.op_graph = Merger().merge([n.value for n in self.networks])

    def __str__(self):
        strs = ["Property:"]
        strs += ["  Networks:"] + ["    " + str(self.networks)]
        strs += ["  Input Constraint:"] + [
            "    " + s for s in str(self.input_constraint).split("\n")
        ]
        strs += ["  Output Constraint:"] + [
            "    " + s for s in str(self.output_constraint).split("\n")
        ]
        return "\n".join(strs)

    def validate_counter_example(
        self, cex: np.ndarray, threshold=1e-6
    ) -> Tuple[bool, Optional[str]]:
        if not self.input_constraint.validate(cex, threshold=threshold):
            return False, "Invalid counter example found: input outside bounds."
        output = self.op_graph(cex)
        if not self.output_constraint.validate(output, threshold=threshold):
            return False, "Invalid counter example found: output outside bounds."
        return True, None

    def prefixed_and_suffixed_op_graph(
        self, return_new_bounds=True, return_prefixes=False
    ) -> Union[
        Tuple[OperationGraph],
        Tuple[OperationGraph, Tuple[np.ndarray, np.ndarray]],
        Tuple[OperationGraph, Tuple[OperationGraph, ...]],
        Tuple[
            OperationGraph, Tuple[np.ndarray, np.ndarray], Tuple[OperationGraph, ...]
        ],
    ]:
        if not isinstance(self.input_constraint, HyperRectangle):
            raise ValueError(
                f"{type(self.input_constraint).__name__} input constraints are not yet supported"
            )

        suffixed_op_graph = self.suffixed_op_graph()

        class PrefixTransformer(OperationTransformer):
            def __init__(self, lbs, ubs):
                super().__init__()
                self.lbs = lbs
                self.ubs = ubs
                self._input_count = 0
                self.final_lbs = []
                self.final_ubs = []
                self.prefix_ops = []

            def visit_Input(
                self, operation: operations.Input
            ) -> Union[operations.Conv, operations.Gemm]:
                dtype = operation.dtype
                new_op: Union[operations.Conv, operations.Gemm]
                input_shape = self.lbs[self._input_count].shape
                if len(input_shape) == 2:
                    ranges = self.ubs[self._input_count] - self.lbs[self._input_count]
                    mins = self.lbs[self._input_count]
                    new_op = operations.Gemm(
                        operation,
                        np.diag(ranges.flatten()).astype(dtype),
                        mins.flatten().astype(dtype),
                    )
                    self.final_lbs.append(np.zeros_like(mins))
                    self.final_ubs.append(np.ones_like(mins))
                elif len(input_shape) == 4:
                    b = self.lbs[self._input_count].min(axis=(0, 2, 3))
                    r = self.ubs[self._input_count].max(axis=(0, 2, 3)) - b
                    c = len(r)
                    w = np.zeros((c, c, 1, 1))
                    for i in range(c):
                        w[i, i, 0, 0] = 1
                    w = w * r.reshape((c, 1, 1, 1))
                    new_op = operations.Conv(
                        operation, w.astype(dtype), b.astype(dtype)
                    )
                    tile_shape = list(self.lbs[self._input_count].shape)
                    tile_shape[1] = 1
                    tiled_b = np.tile(b.reshape((1, -1, 1, 1)), tile_shape)
                    tiled_r = np.tile(r.reshape((1, -1, 1, 1)), tile_shape)
                    self.final_lbs.append(
                        (self.lbs[self._input_count] - tiled_b) / tiled_r
                    )
                    self.final_ubs.append(
                        (self.ubs[self._input_count] - tiled_b) / tiled_r
                    )
                else:
                    raise NotImplementedError(
                        f"Cannot prefix network with input shape {input_shape}"
                    )
                self._input_count += 1
                self.prefix_ops.append(new_op)
                return new_op

        prefix_transformer = PrefixTransformer(
            self.input_constraint.lower_bounds,
            self.input_constraint.upper_bounds,
        )
        prefixed_op_graph = OperationGraph(suffixed_op_graph.walk(prefix_transformer))
        if return_new_bounds and return_prefixes:
            return (
                prefixed_op_graph,
                (prefix_transformer.final_lbs, prefix_transformer.final_ubs),
                tuple(OperationGraph([op]) for op in prefix_transformer.prefix_ops),
            )
        if return_new_bounds:
            return (
                prefixed_op_graph,
                (prefix_transformer.final_lbs, prefix_transformer.final_ubs),
            )
        if return_prefixes:
            return prefixed_op_graph, tuple(
                OperationGraph([op]) for op in prefix_transformer.prefix_ops
            )
        return (prefixed_op_graph,)

    def suffixed_op_graph(self) -> OperationGraph:
        op_graph = self.op_graph.copy()
        if len(op_graph.output_operations) == 1:
            new_output_op = op_graph.output_operations[0]
        else:
            output_operations = [
                operations.Flatten(o) for o in op_graph.output_operations
            ]
            new_output_op = operations.Concat(output_operations, axis=1)
        dtype = OperationGraph([new_output_op]).output_details[0].dtype
        size = self.output_constraint.size()
        k = len(self.output_constraint.halfspaces)
        W = np.zeros((size, k), dtype=dtype)
        b = np.zeros(k, dtype=dtype)
        for n, hs in enumerate(self.output_constraint.halfspaces):
            b[n] = -hs.b
            if hs.is_open:
                b[n] += 1e-6  # TODO : remove magic number
            for i, c in zip(hs.indices, hs.coefficients):
                W[i, n] = c
        new_output_op = operations.Gemm(new_output_op, W, b)
        new_output_op = operations.Relu(new_output_op)

        W_mask = np.zeros((k, 1), dtype=dtype)
        for i in range(k):
            W_mask[i, 0] = 1
        new_output_op = operations.Gemm(new_output_op, W_mask, np.zeros(1, dtype=dtype))
        return OperationGraph([new_output_op])


__all__ = [
    "IOPolytopeProperty",
]
