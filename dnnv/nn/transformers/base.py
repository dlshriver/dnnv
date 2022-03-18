from __future__ import annotations

import inspect

from typing import Iterable, List

from ..graph import OperationGraph
from ..operations import Operation


class OperationTransformer:
    def __init__(self, cached=True, recursive=True):
        self.cached = cached
        self.recursive = recursive
        self._cache = {}

    def generic_visit(self, operation: Operation) -> Operation:
        parameters = {}
        make_new_op = False
        for name in inspect.signature(type(operation)).parameters:
            value = getattr(operation, name)
            if isinstance(value, Operation):
                visited_value = self.visit(value)
                if visited_value is not value:
                    make_new_op = True
            elif isinstance(value, (list, set, tuple)):
                new_value = []
                for value_ in value:
                    if isinstance(value_, Operation):
                        visited_value_ = self.visit(value_)
                        new_value.append(visited_value_)
                        if visited_value_ is not value_:
                            make_new_op = True
                    else:
                        new_value.append(value_)
                visited_value = type(value)(new_value)
            else:
                visited_value = value
            parameters[name] = visited_value
        if make_new_op:
            return type(operation)(**parameters)
        return operation

    def transform(self, op_graph: OperationGraph) -> OperationGraph:
        outputs = []
        for op in op_graph.outputs:
            transformed_op = self.visit(op)
            if isinstance(transformed_op, Iterable):
                outputs.extend(transformed_op)
            else:
                outputs.append(transformed_op)
        return OperationGraph(outputs)

    def visit(self, operation: Operation):
        if self.cached and operation in self._cache:
            return self._cache[operation]
        if self.recursive:
            return self._recursive_visit(operation)
        return self._visit(operation)

    def _recursive_visit(self, operation: Operation):
        method_name = f"visit_{operation}"
        visitor = getattr(self, method_name, self.generic_visit)
        self._cache[operation] = visitor(operation)
        return self._cache[operation]

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
            if not self.cached or operation not in self._cache:
                method_name = f"visit_{operation}"
                visitor = getattr(self, method_name, self.generic_visit)
                self._cache[operation] = visitor(operation)
        return self._cache[_operation]


__all__ = ["OperationTransformer"]
