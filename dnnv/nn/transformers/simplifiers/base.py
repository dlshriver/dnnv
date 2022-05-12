from __future__ import annotations

from typing import List, Type

from ...operations import Operation
from ..base import OperationTransformer


class Simplifier(OperationTransformer):
    def __init__(self):
        super().__init__(cached=True, recursive=False)

    def simplify(self, operation: Operation) -> Operation:
        if hasattr(self, f"visit_{type(operation)}"):
            return getattr(self, f"visit_{type(operation)}")(operation)
        return operation

    def transform(self, op_graph):
        simplified_op_graph = super().transform(op_graph)
        simplified_op_graph._inputs = op_graph._inputs
        simplified_op_graph._input_details = op_graph._input_details
        simplified_op_graph._output_details = op_graph._output_details
        return simplified_op_graph

    def _visit(self, operation: Operation):
        _operation = operation
        stack: List[Operation] = [operation]
        visit_stack = []
        while len(stack):
            operation = stack.pop()
            visit_stack.append(operation)
            for input_op in operation.inputs:
                stack.append(input_op)
        while len(visit_stack):
            operation = visit_stack.pop()
            if operation not in self._cache:
                result = self.generic_visit(operation)
                result = self.simplify(result)
                self._cache[operation] = result
        return self._cache[_operation]


class ComposeSimplifiers(Simplifier):
    def __init__(self, *simplifiers: Type[Simplifier]):
        super().__init__()
        self.simplifiers = [simplifier() for simplifier in simplifiers]

    def simplify(self, operation: Operation) -> Operation:
        modified_graph = True
        while modified_graph:
            modified_graph = False
            for simplifier in self.simplifiers:
                result = simplifier.simplify(operation)
                if result is not operation:
                    modified_graph = True
                operation = result
        return operation


__all__ = ["Simplifier", "ComposeSimplifiers"]
