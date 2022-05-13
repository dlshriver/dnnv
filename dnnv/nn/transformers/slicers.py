from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple, Union

from ..graph import OperationGraph
from ..operations import Input, Operation
from .base import OperationTransformer


class Slicer(OperationTransformer):
    # TODO : This class needs a refactor
    def __init__(self, start: Optional[int], stop: Optional[int]):
        super().__init__(cached=False)
        self.start = start or 0
        self.stop = stop

        self.index = 0
        self.length = 0
        self._index_cache: Dict[
            Operation, Tuple[Union[float, int], Union[float, int], Operation]
        ] = {}
        self.current_pass: Optional[str] = None

    def visit(self, operation: Operation) -> Union[Operation, Sequence[Operation]]:
        is_first = False
        if self.current_pass is None:
            is_first = True

        if is_first:
            self._cache = {}

            # compute indices of every node
            self.current_pass = "indexing"
            super().visit(operation)

            # select output and input nodes
            self.current_pass = "selection"
            outputs = []
            if self.stop is None:
                outputs.append(super().visit(operation))
            elif self.stop == 0:
                outputs = []
            else:
                for pi, ni, op in self._index_cache.values():
                    if min(self.stop, self.length) in (pi + 1, ni + 1):
                        outputs.append(super().visit(op))

            # reset pass
            self.current_pass = None
            return outputs
        elif self.current_pass == "indexing":
            if operation in self._cache:
                raise ValueError("Slicing cyclic graphs is not supported.")
            self._cache[operation] = None
            super().visit(operation)
            del self._cache[operation]
            return operation
        elif self.current_pass == "selection":
            if operation not in self._cache:
                self._cache[operation] = super().visit(operation)
            return self._cache[operation]
        else:
            raise ValueError()

    def generic_visit(self, operation: Operation) -> Operation:
        if self.current_pass == "indexing":
            self.index -= 1
            self.length = max(self.length, -self.index)
            if operation not in self._index_cache:
                self._index_cache[operation] = (
                    float("inf"),
                    float("-inf"),
                    operation,
                )
            pos_index, neg_index, op = self._index_cache[operation]
            self._index_cache[operation] = (
                pos_index,
                max(neg_index, self.index),
                op,
            )
            result = super().generic_visit(operation)
            pos_index, neg_index, op = self._index_cache[operation]
            self._index_cache[operation] = (
                min(pos_index, self.length + self.index),
                neg_index,
                op,
            )
            self.index += 1
            return result
        elif self.current_pass == "selection":
            pos_index, neg_index, _ = self._index_cache[operation]
            if (self.start > 0 and pos_index < self.start) or (
                self.start < 0 and neg_index < self.start
            ):
                y = OperationGraph([operation]).output_details
                return Input(y[0].shape, y[0].dtype)
            return super().generic_visit(operation)
        else:
            raise ValueError(f"Unknown slicing pass: {self.current_pass}")


__all__ = ["Slicer"]
