from typing import Set

from .. import Operation, operations
from ..graph import OperationGraph
from .base import Analysis


class SplitAnalysis(Analysis):
    def __init__(self, dnn: OperationGraph):
        self._cache: Set[Operation] = set()
        super().__init__(dnn)

    def visit(self, operation: Operation):
        if operation not in self._cache:
            self._cache.add(operation)
            super().visit(operation)
            self.results[operation] = False
        else:
            self.results[operation] = True
        return self


class ConstantsAnalysis(Analysis):
    def visit(self, operation: Operation):
        super().visit(operation)
        if operation in self.results:
            return self
        for value in operation.__dict__.values():
            if isinstance(value, Operation) and not self.results[value]:
                self.results[operation] = False
                break
        if operation not in self.results:
            self.results[operation] = True
        return self

    def visit_Input(self, operation: operations.Input):
        self.results[operation] = False

    def visit_Shape(self, operation: operations.Shape):
        shape = OperationGraph([operation.x]).output_shape[0]
        if all(d >= 0 for d in shape):
            self.results[operation] = True


__all__ = ["Analysis", "SplitAnalysis"]
