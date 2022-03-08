from __future__ import annotations

from ..graph import OperationGraph
from ..operations import Operation


class OperationVisitor:
    def __init__(self, cached=True):
        self.cached = cached
        self.visited = set()

    def visit(self, operation: Operation):
        if not self.cached or operation not in self.visited:
            self.visited.add(operation)
            method_name = "visit_%s" % operation.__class__.__name__
            visitor = getattr(self, method_name, self.generic_visit)
            visitor(operation)

    def generic_visit(self, operation: Operation):
        for value in operation.__dict__.values():
            if isinstance(value, Operation):
                self.visit(value)
            elif isinstance(value, (list, tuple)):
                for sub_value in value:
                    if isinstance(sub_value, Operation):
                        self.visit(sub_value)

    def walk(self, op_graph: OperationGraph):
        for op in op_graph.outputs:
            self.visit(op)


__all__ = ["OperationVisitor"]
