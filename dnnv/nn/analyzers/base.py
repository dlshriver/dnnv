from typing import Any, Dict

from .. import Operation
from ..graph import OperationGraph
from ..visitors import OperationVisitor


class Analysis(OperationVisitor):
    def __init__(self, dnn: OperationGraph):
        self.results: Dict[Operation, Any] = {}
        dnn.walk(self)

    def __getitem__(self, index):
        return self.results[index]


__all__ = ["Analysis"]
