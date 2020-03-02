import itertools
import numpy as np

from collections import namedtuple
from typing import List

from .operations import Input, Operation


class OperationGraph:
    def __init__(self, output_operations: List[Operation]):
        self.output_operations = tuple(output_operations)

    def walk(self, visitor):
        result = []
        for output_operation in self.output_operations:
            result.append(visitor.visit(output_operation))
        return result

    def copy(self):
        return self[:]

    def simplify(self):
        from .transformers import simplify

        return simplify(self)

    def pprint(self):
        from .visitors import PrintVisitor

        return self.walk(PrintVisitor())

    @property
    def input_details(self):
        from .visitors import GetInputDetails

        return tuple(itertools.chain.from_iterable(self.walk(GetInputDetails())))

    @property
    def input_shape(self):
        return tuple(details.shape for details in self.input_details)

    @property
    def output_details(self):
        OutputDetails = namedtuple("OutputDetails", ["shape", "dtype"])
        output = self(
            *[
                np.ones([i if i >= 0 else 1 for i in d.shape], dtype=d.dtype)
                for d in self.input_details
            ],
            squeeze=False,
        )
        return tuple(OutputDetails(o.shape, o.dtype) for o in output)

    @property
    def output_shape(self):
        return tuple(details.shape for details in self.output_details)

    @property
    def is_linear(self) -> bool:
        from .visitors import OperationVisitor

        LinearPattern = (Operation >> Operation) | Input

        class IsLinear(OperationVisitor):
            def __init__(self):
                super().__init__()
                self.result = True

            def generic_visit(self, operation):
                matches = LinearPattern.match([operation])
                found_match = False
                for match in matches:
                    found_match = True
                    self.result &= len(match) <= 1
                    if not self.result:
                        continue
                self.result &= found_match
                if not self.result:
                    return False
                return super().generic_visit(operation)

        linearity_visitor = IsLinear()
        self.walk(linearity_visitor)
        result = linearity_visitor.result
        return result

    def as_tf(self):
        from .converters.tensorflow import convert as tf_convert

        return tf_convert(self.output_operations)

    def __call__(self, *x, **kwargs):
        tf_func = self.as_tf()
        output = tf_func(*x, **kwargs)
        return output

    def __getitem__(self, index):
        if isinstance(index, slice):
            index = (index,)
        elif isinstance(index, int):
            index = (slice(None), index)
        elif not isinstance(index, tuple):
            raise TypeError(
                f"Unsupported type for indexing operation graph: {type(index).__name__!r}"
            )
        elif len(index) != 2:
            raise TypeError(f"Unsupported indexing expression {index!r}")
        elif not isinstance(index[0], slice):
            raise TypeError(
                f"Unsupported type for slicing indices: {type(index[0]).__name__!r}"
            )
        elif not isinstance(index[1], int):
            raise TypeError(
                f"Unsupported type for selecting operations: {type(index[1]).__name__!r}"
            )

        if index[0].step is not None:
            raise ValueError("Slicing does not support non-unit steps.")
        start = index[0].start or 0
        stop = index[0].stop

        from .transformers import Slicer

        result = list(itertools.chain.from_iterable(self.walk(Slicer(start, stop))))
        if len(index) > 1:
            output_select = index[1]
            if isinstance(output_select, int):
                output_select = [output_select]
            result = [result[i] for i in output_select]
        sliced_op_graph = OperationGraph(result)
        return sliced_op_graph
