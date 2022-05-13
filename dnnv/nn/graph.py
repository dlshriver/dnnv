from __future__ import annotations

import itertools
from typing import Dict, List, Optional, Sequence, Tuple

from . import operations
from .operations import Input, Operation
from .utils import TensorDetails
from .visitors import PrintVisitor


class OperationGraph:
    def __init__(self, outputs: Sequence[Operation]):
        self.outputs: Sequence[Operation] = tuple(outputs)

        self._inputs: Optional[Sequence[Input]] = None
        self._input_details: Optional[Sequence[TensorDetails]] = None
        self._output_details: Optional[Sequence[TensorDetails]] = None
        self._is_linear: Optional[bool] = None
        self._simplified: Optional[OperationGraph] = None
        self._tensor_details: Optional[Dict[Operation, TensorDetails]] = None
        self._generators: Optional[Sequence[operations.RandomUniform]] = None

    @property
    def tensor_details(self) -> Dict[Operation, TensorDetails]:
        if self._tensor_details is None:
            from .visitors.inference import DetailsInference

            inference_visitor = DetailsInference()
            inference_visitor.walk(self)
            self._tensor_details = inference_visitor.tensor_details
        return self._tensor_details

    @property
    def inputs(self) -> Sequence[Input]:
        if self._inputs is None:
            from .visitors import OperationVisitor

            inputs: List[Input] = []

            class GetInputs(OperationVisitor):
                def visit_Input(self, op: Input):
                    nonlocal inputs
                    inputs.append(op)

            GetInputs().walk(self)
            self._inputs = tuple(inputs)

        return self._inputs

    @property
    def generators(self) -> Sequence[operations.RandomUniform]:
        if self._generators is None:
            from .visitors import OperationVisitor

            generators: List[operations.RandomUniform] = []

            class GetGenerators(OperationVisitor):
                def visit_RandomUniform(self, op: operations.RandomUniform):
                    nonlocal generators
                    generators.append(op)

            GetGenerators().walk(self)
            self._generators = tuple(generators)

        return self._generators

    @property
    def input_details(self) -> Sequence[TensorDetails]:
        if self._input_details is None:
            from .utils import TensorDetails

            input_details = []
            for input_op in self.inputs:
                input_details.append(
                    TensorDetails(tuple(int(i) for i in input_op.shape), input_op.dtype)
                )
            self._input_details = tuple(input_details)
        return self._input_details

    @property
    def input_shape(self) -> Sequence[Tuple[int, ...]]:
        return tuple(details.shape for details in self.input_details)

    @property
    def output_details(self) -> Sequence[TensorDetails]:
        if self._output_details is None:
            self._output_details = tuple(self.tensor_details[o] for o in self.outputs)
        return self._output_details

    @property
    def output_shape(self) -> Sequence[Tuple[int, ...]]:
        return tuple(details.shape for details in self.output_details)

    def as_onnx(self, **kwargs):
        from .converters.onnx import OnnxConverter

        return OnnxConverter.convert(self, **kwargs)

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
                super().__init__(recursive=False)
                self.input_op_graph = input_op_graph
                self.input_id = 0
                self.visited: Dict[Operation, Operation] = {}

            def visit_Input(self, operation: operations.Input):
                new_op = self.input_op_graph.outputs[self.input_id]
                self.input_id += 1
                return new_op

        new_op_graph = Composer(input_op_graph).transform(self)
        new_op_graph._inputs = input_op_graph._inputs
        new_op_graph._input_details = input_op_graph.input_details
        new_op_graph._output_details = self.output_details

        return new_op_graph

    def copy(self) -> OperationGraph:
        from .transformers.copy import Copier

        copier = Copier()
        return copier.transform(self)

    def export_onnx(self, filename, **kwargs):
        import onnx

        onnx.save(self.as_onnx(**kwargs), filename)

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
            linearity_visitor.walk(self)
            self._is_linear = result
        return self._is_linear

    def pprint(self):
        return PrintVisitor().walk(self)

    def simplify(self) -> OperationGraph:
        if self._simplified is None:
            from .transformers import simplify

            result = simplify(self)
            self._simplified = result
            result._simplified = result
        return self._simplified

    def __call__(self, *x, _squeeze=True, **kwargs):
        assert len(kwargs) == 0
        import onnxruntime.backend

        onnx_model = self.as_onnx(**kwargs)
        y = onnxruntime.backend.run(onnx_model, list(x))
        if _squeeze and len(y) == 1:
            return y[0]
        return y

    def __getitem__(self, index):
        from .transformers import Slicer

        if isinstance(index, slice):
            index = (index, list(range(len(self.outputs))))
        elif isinstance(index, int):
            index = (slice(None), [index])
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

        output_select = index[1]
        if isinstance(output_select, int):
            output_select = [output_select]

        if start is None or start == 0 and stop is None:
            copy = self.copy()
            output_ops = copy.outputs
            sliced_op_graph = OperationGraph([output_ops[i] for i in output_select])
            return sliced_op_graph

        from .transformers import Slicer

        result = list(
            itertools.chain.from_iterable(
                Slicer(start, stop).visit(op) for op in self.outputs
            )
        )
        result = [result[i] for i in output_select]
        sliced_op_graph = OperationGraph(result)
        return sliced_op_graph


__all__ = ["OperationGraph"]
