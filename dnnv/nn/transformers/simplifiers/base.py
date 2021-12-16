from typing import Any, Dict, List, Optional, Set, Type, Union

from ..base import OperationTransformer
from ...analyzers import Analysis
from ...graph import OperationGraph
from ...operations import Operation


class Simplifier(OperationTransformer):
    ANALYSES: Dict[str, Type[Analysis]] = {}

    def __init__(self, dnn: OperationGraph):
        super().__init__()
        self._modified_graph = False
        self.dnn = dnn
        self.run_analyses()

    def run_analyses(self):
        self.analysis: Dict[str, Analysis] = {
            name: analysis(self.dnn) for name, analysis in self.ANALYSES.items()
        }

    def simplify(self, operation: Operation) -> Operation:
        if hasattr(self, f"visit_{type(operation)}"):
            return getattr(self, f"visit_{type(operation)}")(operation)
        return operation

    def visit(self, operation: Operation) -> Operation:
        if operation not in self._cache:
            operation = super().generic_visit(operation)
            result = self.simplify(operation)
            if result is not operation:
                self._modified_graph = True
            self._cache[operation] = result
        return self._cache[operation]


class ComposeSimplifiers(Simplifier):
    def __init__(self, dnn: OperationGraph, *simplifiers: Type[Simplifier]):
        super().__init__(dnn)
        self.simplifiers = [simplifier(dnn) for simplifier in simplifiers]

    def visit(self, operation: Operation) -> Operation:
        if operation in self._cache:
            return self._cache[operation]
        operation = super().generic_visit(operation)
        modified_graph = True
        while modified_graph:
            modified_graph = False
            for simplifier in self.simplifiers:
                simplifier._modified_graph = False
                result = simplifier.simplify(operation)
                if result is not operation:
                    modified_graph = True
                    for simplifier in self.simplifiers:
                        simplifier.run_analyses()
                operation = result
            self._modified_graph |= modified_graph
        self._cache[operation] = operation
        return operation


__all__ = ["Simplifier", "ComposeSimplifiers"]
