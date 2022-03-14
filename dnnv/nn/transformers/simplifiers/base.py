from __future__ import annotations

from typing import Type

from ..base import OperationTransformer
from ...graph import OperationGraph
from ...operations import Operation


class Simplifier(OperationTransformer):
    def __init__(self, dnn: OperationGraph):
        super().__init__()
        self.dnn = dnn

    def simplify(self, operation: Operation) -> Operation:
        if hasattr(self, f"visit_{type(operation)}"):
            return getattr(self, f"visit_{type(operation)}")(operation)
        return operation

    def visit(self, operation: Operation) -> Operation:
        if operation not in self._cache:
            operation = super().generic_visit(operation)
            result = self.simplify(operation)
            self._cache[operation] = result
        return self._cache[operation]


class ComposeSimplifiers(Simplifier):
    def __init__(self, dnn: OperationGraph, *simplifiers: Type[Simplifier]):
        super().__init__(dnn)
        self.simplifiers = [simplifier(dnn) for simplifier in simplifiers]

    def visit(self, operation: Operation) -> Operation:
        if operation in self._cache:
            return self._cache[operation]
        operation = super().generic_visit(operation)
        modified_graph = True
        while modified_graph:
            modified_graph = False
            for simplifier in self.simplifiers:
                result = simplifier.simplify(operation)
                if result is not operation:
                    modified_graph = True
                operation = result
        self._cache[operation] = operation
        return operation


__all__ = ["Simplifier", "ComposeSimplifiers"]
