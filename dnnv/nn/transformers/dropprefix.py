from __future__ import annotations

import numpy as np

from copy import deepcopy

from ..graph import OperationGraph
from ..operations import Input, Operation

from .base import OperationTransformer


class DropPrefix(OperationTransformer):
    def __init__(self, prefix_graph: OperationGraph):
        super().__init__()
        self.prefix_graph = prefix_graph

    def visit(self, operation: Operation) -> Operation:
        if operation not in self._cache:
            if operation not in self.prefix_graph.outputs:
                result = super().visit(operation)
            else:
                y = OperationGraph([operation]).output_details
                if len(y) > 1:
                    raise ValueError(
                        "Dropping prefixes with multiple output values is currently not supported"
                    )
                new_input_shape = np.asarray(y[0].shape)
                result = Input(new_input_shape, y[0].dtype)
            self._cache[operation] = result
        return self._cache[operation]

    def generic_visit(self, operation: Operation) -> Operation:
        kwargs = {}
        for name, value in operation.__dict__.items():
            if isinstance(value, Operation):
                new_value = self.visit(value)
                kwargs[name] = new_value
            else:
                kwargs[name] = deepcopy(value)
        return operation.__class__(**kwargs)


__all__ = ["DropPrefix"]
