from __future__ import annotations

from typing import Sequence, Type

from .. import Operation
from .base import OperationVisitor


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


__all__ = ["EnsureSupportVisitor"]
