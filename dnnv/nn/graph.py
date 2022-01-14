from __future__ import annotations

import itertools
import numpy as np

from typing import Dict, Optional, Sequence, Tuple

from .operations import Operation
from .utils import TensorDetails


class OperationGraph:
    def __init__(self, output_operations: Sequence[Operation]):
        self.output_operations = tuple(output_operations)
        self._input_details: Optional[Sequence[TensorDetails]] = None
        self._output_details: Optional[Sequence[TensorDetails]] = None
        self._is_linear: Optional[bool] = None

    def walk(self, visitor):
        result = []
        for output_operation in self.output_operations:
            result.append(visitor.visit(output_operation))
        return result

    def copy(self) -> OperationGraph:
        from .transformers.copy import Copier

        copier = Copier()
        return copier.transform(self)

    def simplify(self) -> OperationGraph:
        from .transformers import simplify

        return simplify(self.copy())

    def pprint(self):
        from .visitors import PrintVisitor

        return self.walk(PrintVisitor())

    def is_linear(self):
        if self._is_linear is None:
            from .operations import Input
            from .visitors import OperationVisitor

            LinearPattern = (Operation >> Operation) | Input
            result = True

            class IsLinear(OperationVisitor):
                def generic_visit(self, operation):
                    nonlocal result

                    matches = LinearPattern.match([operation])
                    found_match = False
                    for match in matches:
                        found_match = True
                        result &= len(match) <= 1
                        if not result:
                            continue
                    result &= found_match
                    if not result:
                        return False
                    return super().generic_visit(operation)

            linearity_visitor = IsLinear()
            self.walk(linearity_visitor)
            self._is_linear = result
        return self._is_linear

    @property
    def input_details(self) -> Sequence[TensorDetails]:
        if self._input_details is None:
            from .visitors import GetInputDetails

            input_details_visitor = GetInputDetails()
            result = ()
            for output_op in self.output_operations:
                result = input_details_visitor.visit(output_op)
            self._input_details = result
        return self._input_details

    @property
    def input_shape(self) -> Sequence[Tuple[int, ...]]:
        return tuple(details.shape for details in self.input_details)

    @property
    def output_details(self) -> Sequence[TensorDetails]:
        # TODO : fix output_details calculation
        # for operations with multiple outputs
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
    def output_shape(self) -> Sequence[Tuple[int, ...]]:
        # TODO : fix output_shape calculation
        # for operations with multiple outputs
        return tuple(details.shape for details in self.output_details)

    def export_onnx(self, filename, *, add_missing_optional_inputs=False):
        import onnx

        onnx.save(
            self.as_onnx(add_missing_optional_inputs=add_missing_optional_inputs),
            filename,
        )

    def as_onnx(self, *, add_missing_optional_inputs=False):
        from .converters.onnx import convert as onnx_convert

        return onnx_convert(
            self, add_missing_optional_inputs=add_missing_optional_inputs
        )

    def as_tf(self):
        from .converters.tensorflow import convert as tf_convert

        return tf_convert(self)

    def compose(self, input_op_graph: OperationGraph) -> OperationGraph:
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
                super().__init__()
                self.input_op_graph = input_op_graph
                self.input_id = 0
                self.visited: Dict[Operation, Operation] = {}

            def visit_Input(self, operation: operations.Input):
                new_op = self.input_op_graph.output_operations[self.input_id]
                self.input_id += 1
                return new_op

            def generic_visit(self, operation: Operation) -> Operation:
                operation = super().generic_visit(operation)
                operation.name = operation.name + ":!"
                return operation

        new_op_graph = OperationGraph(self.walk(Composer(input_op_graph)))
        new_op_graph._input_details = input_op_graph.input_details
        new_op_graph._output_details = self.output_details

        return new_op_graph

    def __call__(self, *x, **kwargs):
        tf_func = self.as_tf()
        output = tf_func(*x, **kwargs)
        return output

    def __getitem__(self, index):
        if isinstance(index, slice):
            index = (index, list(range(len(self.output_operations))))
        elif isinstance(index, int):
            index = (slice(None), [index])
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

        if index[0].step is not None and index[0].step != 1:
            raise ValueError("Slicing does not support non-unit steps.")
        start = index[0].start or 0
        stop = index[0].stop

        output_select = index[1]
        if isinstance(output_select, int):
            output_select = [output_select]

        if start is None or start == 0 and stop is None:
            copy = self.copy()
            output_ops = copy.output_operations
            sliced_op_graph = OperationGraph([output_ops[i] for i in output_select])
            return sliced_op_graph

        from .transformers import Slicer

        result = list(itertools.chain.from_iterable(self.walk(Slicer(start, stop))))
        result = [result[i] for i in output_select]
        sliced_op_graph = OperationGraph(result)
        return sliced_op_graph


__all__ = ["OperationGraph"]
