from __future__ import annotations

import typing
from typing import Sequence, Type

from .base import OperationVisitor

if typing.TYPE_CHECKING:
    from .. import Operation, OperationGraph


class EnsureSupportVisitor(OperationVisitor):
    def __init__(
        self,
        supported_operations: Sequence[Type[Operation]],
        error_type: Type[Exception] = RuntimeError,
    ):
        super().__init__()
        self.supported = set(supported_operations)
        self.error_type = error_type

    def visit(self, operation: Operation):
        if type(operation) not in self.supported:
            raise self.error_type(f"{type(operation)} operations are not supported.")
        super().visit(operation)

    def walk(self, op_graph: OperationGraph) -> bool:
        super().walk(op_graph)
        return True


__all__ = ["EnsureSupportVisitor"]
