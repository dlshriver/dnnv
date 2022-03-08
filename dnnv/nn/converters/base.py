from __future__ import annotations

from typing import Any

from .. import operations
from ..graph import OperationGraph


class OperationConverter:
    def __init__(self):
        self._cache = {}

    def visit(self, operation: operations.Operation) -> Any:
        if operation not in self._cache:
            method_name = "visit_%s" % operation.__class__.__name__
            visitor = getattr(self, method_name, self.generic_visit)
            self._cache[operation] = visitor(operation)
        return self._cache[operation]

    def generic_visit(self, operation: operations.Operation):
        raise ValueError(
            f"{type(self).__name__} not implemented for operation type {type(operation).__name__}"
        )

    def convert(self, op_graph: OperationGraph) -> Any:
        raise NotImplementedError()
