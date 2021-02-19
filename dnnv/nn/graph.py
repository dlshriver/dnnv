import itertools
import numpy as np

from typing import List, Set

from .operations import Input, Operation
from .utils import TensorDetails


class OperationGraph:
    def __init__(self, output_operations: List[Operation]):
        self.output_operations = tuple(output_operations)
        self._input_details = None
        self._output_details = None

    def walk(self, visitor):
        result = []
        for output_operation in self.output_operations:
            result.append(visitor.visit(output_operation))
        return result

    def copy(self):
        clone = self[:]
        clone._input_details = self._input_details
        clone._output_details = self._output_details
        return clone

    def simplify(self):
        from .transformers import simplify

        return simplify(self.copy())

    def pprint(self):
        from .visitors import PrintVisitor

        return self.walk(PrintVisitor())

    @property
    def input_details(self):
        if self._input_details is None:
            from .visitors import GetInputDetails

            self._input_details = tuple(
                itertools.chain.from_iterable(self.walk(GetInputDetails()))
            )
        return self._input_details

    @property
    def input_shape(self):
        return tuple(details.shape for details in self.input_details)

    @property
    def output_details(self):
        if self._output_details is None:
            output = self(
                *[
                    np.ones([i if i >= 0 else 1 for i in d.shape], dtype=d.dtype)
                    for d in self.input_details
                ],
                squeeze=False,
            )
            self._output_details = tuple(
                TensorDetails(o.shape, o.dtype) for o in output
            )
        return self._output_details

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

    def export_onnx(self, filename):
        import onnx

        onnx.save(self.as_onnx(), filename)

    def as_onnx(self):
        from .converters.onnx import convert as onnx_convert

        return onnx_convert(self)

    def as_pytorch(self):
        from .converters.pytorch import convert as pytorch_convert

        return pytorch_convert(self)

    def as_tf(self):
        from .converters.tensorflow import convert as tf_convert

        return tf_convert(self)

    def compose(self, input_op_graph: "OperationGraph") -> "OperationGraph":
        from .transformers import OperationTransformer
        from . import operations

        self_input_details = self.input_details
        input_op_graph_output_details = input_op_graph.output_details
        if len(self_input_details) != len(input_op_graph_output_details):
            raise ValueError(
                "Number of inputs and outputs must match for op graph composition."
            )
        if any(
            tuple(int(d) if d > 0 else 1 for d in in_details.shape)
            != tuple(int(d) if d > 0 else 1 for d in out_details.shape)
            or in_details.dtype != out_details.dtype
            for in_details, out_details in zip(
                self_input_details, input_op_graph_output_details
            )
        ):
            raise ValueError(
                "Input and output shapes and types must match for op graph composition."
            )

        class Composer(OperationTransformer):
            def __init__(self, input_op_graph: OperationGraph):
                self.input_op_graph = input_op_graph
                self.input_id = 0
                self.visited: Set[Operation] = set()

            def visit(self, operation: Operation):
                if operation not in self.visited:
                    result = super().visit(operation)
                    self.visited.add(operation)
                    return result
                return operation

            def visit_Input(self, operation: operations.Input):
                new_op = self.input_op_graph.output_operations[self.input_id]
                self.input_id += 1
                return new_op

        new_op_graph = OperationGraph(self.copy().walk(Composer(input_op_graph.copy())))

        return new_op_graph

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
