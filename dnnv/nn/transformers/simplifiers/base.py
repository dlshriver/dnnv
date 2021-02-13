from typing import Any, Dict, List, Optional, Set, Type, Union

from ..base import OperationTransformer
from ...analyzers import Analysis
from ...graph import OperationGraph
from ...operations import Operation


class Simplifier(OperationTransformer):
    ANALYSES: Dict[str, Type[Analysis]] = {}

    def __init__(self, dnn: OperationGraph):
        self._cache: Dict[Operation, Operation] = {}
        self._modified_graph = False
        self.dnn = dnn
        self.run_analyses()

    def run_analyses(self):
        self.analysis: Dict[str, Analysis] = {
            name: analysis(self.dnn) for name, analysis in self.ANALYSES.items()
        }

    def visit(self, operation: Operation) -> Operation:
        if operation not in self._cache:
            operation = super().generic_visit(operation)
            result = super().visit(operation)
            if result is not operation:
                self._modified_graph = True
            self._cache[operation] = result
        return self._cache[operation]


class ComposeSimplifiers(Simplifier):
    def __init__(self, dnn: OperationGraph, *simplifiers: Type[Simplifier]):
        super().__init__(dnn)
        self.simplifiers = [simplifier(dnn) for simplifier in simplifiers]

    def visit(self, operation: Operation) -> Operation:
        modified_graph = True
        while modified_graph:
            modified_graph = False
            for simplifier in self.simplifiers:
                simplifier._modified_graph = False
                simplifier._cache = {}
                operation = simplifier.visit(operation)
                modified_graph |= simplifier._modified_graph
                if modified_graph:
                    for simplifier in self.simplifiers:
                        simplifier.run_analyses()
            self._modified_graph |= modified_graph
        return operation


__all__ = ["Simplifier", "ComposeSimplifiers"]
