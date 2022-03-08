from typing import Any, Dict

from .. import Operation
from ..graph import OperationGraph
from ..visitors import OperationVisitor


class Analysis(OperationVisitor):
    def __init__(self, op_graph: OperationGraph):
        super().__init__(cached=False)
        self.results: Dict[Operation, Any] = {}
        self.walk(op_graph)

    def __getitem__(self, index):
        return self.results[index]


__all__ = ["Analysis"]
