from __future__ import annotations

import inspect

from ..graph import OperationGraph
from ..operations import Operation


class OperationTransformer:
    def __init__(self, cached=True):
        self.cached = cached
        self._cache = {}

    def transform(self, op_graph: OperationGraph) -> OperationGraph:
        return OperationGraph([self.visit(op) for op in op_graph.outputs])

    def visit(self, operation: Operation):
        if not self.cached or operation not in self._cache:
            method_name = "visit_%s" % operation.__class__.__name__
            visitor = getattr(self, method_name, self.generic_visit)
            result = visitor(operation)
            if self.cached:
                self._cache[operation] = result
        elif operation in self._cache:
            result = self._cache[operation]
        return result

    def generic_visit(self, operation: Operation) -> Operation:
        parameters = {}
        for name in inspect.signature(type(operation)).parameters:
            value = getattr(operation, name)
            if isinstance(value, Operation):
                value = self.visit(value)
            elif isinstance(value, (list, set, tuple)):
                new_value = []
                for value_ in value:
                    if isinstance(value_, Operation):
                        new_value_ = self.visit(value_)
                        new_value.append(new_value_)
                    else:
                        new_value.append(value_)
                value = type(value)(new_value)
            parameters[name] = value
        return type(operation)(**parameters)


__all__ = ["OperationTransformer"]
