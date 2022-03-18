from __future__ import annotations

import inspect

from .base import OperationTransformer
from ..operations import Operation


class Copier(OperationTransformer):
    def __init__(self):
        super().__init__(recursive=False)

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

    def transform(self, op_graph):
        op_graph_copy = super().transform(op_graph)
        op_graph_copy._inputs = op_graph._inputs
        op_graph_copy._input_details = op_graph._input_details
        op_graph_copy._output_details = op_graph._output_details
        return op_graph_copy


__all__ = ["Copier"]
