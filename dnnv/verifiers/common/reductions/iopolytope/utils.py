from __future__ import annotations

from typing import Dict, Hashable, List

from dnnv.nn import operations
from dnnv.nn.graph import OperationGraph
from dnnv.nn.operations import Operation
from dnnv.nn.transformers import OperationTransformer


class OpGraphMerger(OperationTransformer):
    def __init__(self, *op_graphs: OperationGraph):
        super().__init__()
        self.op_graphs = op_graphs
        self.output_operations: List[Operation] = []

        inputs = op_graphs[0].inputs
        for op_graph in op_graphs[1:]:
            assert inputs == op_graph.inputs

    def merge(self) -> OperationGraph:
        for op_graph in self.op_graphs:
            for op in op_graph.outputs:
                self.output_operations.append(self.visit(op))
        if len(self.output_operations) == 1:
            output_op = self.output_operations[0]
        elif len(self.output_operations) > 1:
            output_operations = [operations.Flatten(o) for o in self.output_operations]
            output_op = operations.Concat(output_operations, axis=1)
        else:
            raise ValueError("No operation graphs to merge")
        return OperationGraph([output_op])


__all__ = ["OpGraphMerger"]
