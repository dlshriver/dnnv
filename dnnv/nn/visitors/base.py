from __future__ import annotations

from typing import List

from ..graph import OperationGraph
from ..operations import Operation


class OperationVisitor:
    def __init__(self, cached=True, recursive=False):
        self.cached = cached
        self.recursive = recursive
        self.visited = set()

    def generic_visit(self, operation: Operation):
        for input_op in operation.inputs:
            self.visit(input_op)

    def visit(self, operation: Operation):
        if self.cached and operation in self.visited:
            return
        if self.recursive:
            self._recursive_visit(operation)
        else:
            self._visit(operation)

    def walk(self, op_graph: OperationGraph):
        for op in op_graph.outputs:
            self.visit(op)

    def _recursive_visit(self, operation: Operation):
        method_name = f"visit_{operation}"
        visitor = getattr(self, method_name, self.generic_visit)
        self.visited.add(operation)
        visitor(operation)

    def _visit(self, operation: Operation):
        stack: List[Operation] = [operation]
        visit_stack = []
        while len(stack):
            operation = stack.pop()
            visit_stack.append(operation)
            for input_op in operation.inputs:
                stack.append(input_op)
        while len(visit_stack):
            operation = visit_stack.pop()
            if not self.cached or operation not in self.visited:
                method_name = f"visit_{operation}"
                visitor = getattr(self, method_name, self.generic_visit)
                self.visited.add(operation)
                visitor(operation)


__all__ = ["OperationVisitor"]
