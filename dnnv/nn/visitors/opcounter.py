from __future__ import annotations

import numpy as np

from .base import OperationVisitor
from .. import Operation, OperationGraph


class OperationCounter(OperationVisitor):
    def visit(self, operation: Operation) -> int:
        super().visit(operation)
        return len(self.visited)

    def walk(self, op_graph: OperationGraph) -> int:
        super().walk(op_graph)
        return len(self.visited)


__all__ = ["OperationCounter"]
