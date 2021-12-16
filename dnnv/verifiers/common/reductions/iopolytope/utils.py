from __future__ import annotations

from typing import Dict, Hashable, List

from dnnv.nn import operations
from dnnv.nn.graph import OperationGraph
from dnnv.nn.operations import Operation
from dnnv.nn.transformers import OperationTransformer
from dnnv.nn.visitors import OperationVisitor


class GetInputs(OperationVisitor):
    def __init__(self):
        super().__init__()
        self.inputs = set()

    def walk(self, op_graph: OperationGraph):
        op_graph.walk(self)
        return self

    def visit_Input(self, op):
        self.inputs.add(op)


class OpGraphMerger(OperationTransformer):
    def __init__(self, *op_graphs: OperationGraph):
        super().__init__()
        self.op_graphs = op_graphs
        self.output_operations: List[Operation] = []
        self.input_operations: Dict[Hashable, Operation] = {}

        inputs = GetInputs().walk(op_graphs[0]).inputs
        for op_graph in op_graphs[1:]:
            assert inputs == GetInputs().walk(op_graph).inputs

    def merge(self) -> OperationGraph:
        for op_graph in self.op_graphs:
            for op in op_graph.output_operations:
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
