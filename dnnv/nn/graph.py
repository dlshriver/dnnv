from __future__ import annotations

import itertools
from typing import Dict, List

import numpy as np
import onnx

from . import operations
from .operations import Operation
from .utils import TensorDetails
from .visitors import GetInputDetails, PrintVisitor


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
        return self.walk(PrintVisitor())

    @property
    def input_details(self):
        if self._input_details is None:
            input_details_visitor = GetInputDetails()
            result = ()
            for output_op in self.output_operations:
                result = input_details_visitor.visit(output_op)
            self._input_details = result
        return self._input_details

    @property
    def input_shape(self):
        return tuple(details.shape for details in self.input_details)

    @property
    def output_details(self):
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
    def output_shape(self):
        # TODO : fix output_shape calculation
        # for operations with multiple outputs
        return tuple(details.shape for details in self.output_details)

    def export_onnx(self, filename, *, add_missing_optional_inputs=False):
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

            def visit_Input(self, _: operations.Input):
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
        from .transformers import Slicer

        if isinstance(index, slice):
            index = (index,)
        elif isinstance(index, int):
            index = (slice(None), index)
        elif not isinstance(index, tuple):
            raise TypeError(
                "Unsupported type for indexing operation graph:"
                f" {type(index).__name__!r}"
            )
        elif len(index) > 2:
            raise TypeError(f"Unsupported indexing expression {index!r}")
        elif not isinstance(index[0], slice):
            raise TypeError(
                f"Unsupported type for slicing indices: {type(index[0]).__name__!r}"
            )
        elif len(index) > 1 and not isinstance(index[1], int):
            raise TypeError(
                "Unsupported type for selecting operations:"
                f" {type(index[1]).__name__!r}"
            )

        if index[0].step is not None and index[0].step != 1:
            raise ValueError("Slicing does not support non-unit steps.")
        start = index[0].start or 0
        stop = index[0].stop

        result = list(itertools.chain.from_iterable(self.walk(Slicer(start, stop))))
        if len(index) > 1:
            output_select = index[1]
            if isinstance(output_select, int):
                output_select = [output_select]
            result = [result[i] for i in output_select]
        sliced_op_graph = OperationGraph(result)
        return sliced_op_graph


__all__ = ["OperationGraph"]
