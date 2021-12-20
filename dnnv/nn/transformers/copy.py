from __future__ import annotations

from .base import OperationTransformer


class Copier(OperationTransformer):
    def transform(self, op_graph):
        op_graph_copy = super().transform(op_graph)
        op_graph_copy._input_details = op_graph._input_details
        op_graph_copy._output_details = op_graph._output_details
        return op_graph_copy


__all__ = ["Copier"]
