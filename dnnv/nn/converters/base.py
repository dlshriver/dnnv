from __future__ import annotations

from typing import Any

from .. import operations
from ..graph import OperationGraph


class OperationConverter:
    def __init__(self):
        self._cache = {}

    def _visit(self, operation: operations.Operation) -> Any:
        stack = [operation]
        visit_stack = []
        while len(stack):
            operation = stack.pop()
            visit_stack.append(operation)
            for child_op in operation.inputs:
                stack.append(child_op)
        while len(visit_stack):
            operation = visit_stack.pop()
            if operation not in self._cache:
                method_name = f"visit_{operation}"
                visitor = getattr(self, method_name, self.generic_visit)
                result = visitor(operation)
                self._cache[operation] = result

    def visit(self, operation):
        if operation not in self._cache:
            self._visit(operation)
        return self._cache[operation]

    def generic_visit(self, operation: operations.Operation):
        raise ValueError(
            f"{type(self).__name__} not implemented for operation type {type(operation).__name__}"
        )

    @classmethod
    def convert(cls, op_graph: OperationGraph) -> Any:
        raise NotImplementedError()
