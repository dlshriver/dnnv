from __future__ import annotations

from typing import List, Type

import numpy as np

from . import operations
from .operations import Operation
from .utils import TensorDetails


class OperationVisitor:
    def visit(self, operation: Operation):
        method_name = f"visit_{type(operation).__name__}"
        visitor = getattr(self, method_name, self.generic_visit)
        return visitor(operation)

    def generic_visit(self, operation: Operation):
        for value in operation.__dict__.values():
            if isinstance(value, Operation):
                self.visit(value)
            elif isinstance(value, (list, tuple)):
                for sub_value in value:
                    if isinstance(sub_value, Operation):
                        self.visit(sub_value)
        return operation


class GetInputDetails(OperationVisitor):
    def __init__(self):
        self.visited = set()
        self.input_details = []

    def visit(self, operation: Operation):
        if id(operation) not in self.visited:
            super().visit(operation)
            self.visited.add(id(operation))
        return tuple(self.input_details)

    def visit_Input(self, operation: operations.Input):
        self.input_details.append(
            TensorDetails(tuple(int(i) for i in operation.shape), operation.dtype)
        )


class OperationCounter(OperationVisitor):
    def __init__(self):
        self.visited = set()

    def visit(self, operation: Operation):
        if id(operation) not in self.visited:
            self.visited.add(id(operation))
            super().generic_visit(operation)
        return len(self.visited)


class EnsureSupportVisitor(OperationVisitor):
    def __init__(
        self,
        supported_operations: List[Type[Operation]],
        error_type: Type[Exception] = RuntimeError,
    ):
        self.supported = set(supported_operations)
        self.error_type = error_type

    def visit(self, operation: Operation):
        if type(operation) not in self.supported:
            raise self.error_type(f"{type(operation)} operations are not supported.")
        return super().visit(operation)


class PrintVisitor(OperationVisitor):
    def __init__(self):
        super().__init__()
        self.visited = set()
        self.identifiers = {"count": {}, "op": {}}

    def visit(self, operation: Operation):
        if operation in self.visited:
            return
        self.visited.add(operation)
        super().visit(operation)

    def generic_visit(self, operation: Operation):
        if not hasattr(self, f"visit_{type(operation).__name__}"):
            raise ValueError(
                f"Operation not currently supported by PrintVisitor: {operation}"
            )
        super().generic_visit(operation)

    def get_op_id(self, operation: Operation) -> str:
        if isinstance(operation, np.ndarray):
            if np.product(operation.shape) < 5:
                return "".join(str(operation).split("\n"))
            return f"ndarray(shape={operation.shape})"
        op_type = operation.__class__.__name__
        if operation not in self.identifiers["op"]:
            idx = self.identifiers["count"].get(op_type, 0)
            self.identifiers["count"][op_type] = idx + 1
            self.identifiers["op"][operation] = idx
        idx = self.identifiers["op"][operation]
        return f"{op_type}_{idx}"

    def print_op_id(self, operation: Operation) -> None:
        op_id = self.get_op_id(operation)
        print(f"{op_id:32s}", end=": ")

    def visit_Add(self, operation: operations.Add) -> None:
        self.generic_visit(operation)
        self.print_op_id(operation)
        print(f"Add({self.get_op_id(operation.a)}, {self.get_op_id(operation.b)})")

    def visit_Atan(self, operation: operations.Atan) -> None:
        self.generic_visit(operation)
        self.print_op_id(operation)
        print(f"Atan({self.get_op_id(operation.x)})")

    def visit_AveragePool(self, operation: operations.AveragePool) -> None:
        self.generic_visit(operation)
        self.print_op_id(operation)
        print(f"AveragePool({self.get_op_id(operation.x)}, {operation.kernel_shape})")

    def visit_BatchNormalization(
        self, operation: operations.BatchNormalization
    ) -> None:
        self.generic_visit(operation)
        self.print_op_id(operation)
        print(f"BatchNormalization({self.get_op_id(operation.x)})")

    def visit_Cast(self, operation: operations.Cast) -> None:
        self.generic_visit(operation)
        self.print_op_id(operation)
        print(f"Cast({self.get_op_id(operation.x)}, to={operation.to})")

    def visit_Concat(self, operation: operations.Concat) -> None:
        self.generic_visit(operation)
        self.print_op_id(operation)
        print(f"Concat({[self.get_op_id(x) for x in operation.x]})")

    def visit_Conv(self, operation: operations.Conv) -> None:
        self.generic_visit(operation)
        self.print_op_id(operation)
        print(
            "Conv("
            f"{self.get_op_id(operation.x)}, "
            f"out_channels={operation.w.shape[0]}, "
            f"kernel_shape={operation.kernel_shape.tolist()}, "
            f"strides={operation.strides.tolist()}, "
            f"pads={operation.pads.tolist()}, "
            f"group={operation.group}, "
            f"dilations={operation.dilations.tolist()}"
            ")",
        )

    def visit_ConvTranspose(self, operation: operations.ConvTranspose) -> None:
        self.generic_visit(operation)
        self.print_op_id(operation)
        print(
            "ConvTranspose("
            f"{self.get_op_id(operation.x)}, "
            f"out_channels={operation.w.shape[1]*operation.group}, "
            f"kernel_shape={operation.kernel_shape.tolist()}, "
            f"strides={operation.strides.tolist()}, "
            f"pads={operation.pads.tolist()}, "
            f"output_padding={operation.output_padding.tolist()}, "
            f"group={operation.group}, "
            f"dilations={operation.dilations.tolist()}"
            ")",
        )

    def visit_Div(self, operation: operations.Div) -> None:
        self.generic_visit(operation)
        self.print_op_id(operation)
        print(f"Div({self.get_op_id(operation.a)}, {self.get_op_id(operation.b)})")

    def visit_Dropout(self, operation: operations.Dropout) -> None:
        self.generic_visit(operation)
        self.print_op_id(operation)
        print(f"Dropout({self.get_op_id(operation.x)}, ratio={operation.ratio})")

    def visit_Elu(self, operation: operations.Elu) -> None:
        self.generic_visit(operation)
        self.print_op_id(operation)
        print(f"Elu({self.get_op_id(operation.x)}, alpha={operation.alpha})")

    def visit_Expand(self, operation: operations.Expand) -> None:
        self.generic_visit(operation)
        self.print_op_id(operation)
        print(
            f"Expand({self.get_op_id(operation.x)}, {self.get_op_id(operation.shape)})"
        )

    def visit_Flatten(self, operation: operations.Flatten) -> None:
        self.generic_visit(operation)
        self.print_op_id(operation)
        print(f"Flatten({self.get_op_id(operation.x)}, axis={operation.axis})")

    def visit_Gather(self, operation: operations.Gather) -> None:
        self.generic_visit(operation)
        self.print_op_id(operation)
        print(
            f"Gather({self.get_op_id(operation.x)},"
            f" {self.get_op_id(operation.indices)}, axis={operation.axis})"
        )

    def visit_Gemm(self, operation: operations.Gemm) -> None:
        self.generic_visit(operation)
        self.print_op_id(operation)
        print(
            "Gemm("
            f"{self.get_op_id(operation.a)}, "
            f"{self.get_op_id(operation.b)}, "
            f"{self.get_op_id(operation.c)}, "
            f"transpose_a={operation.transpose_a:d}, "
            f"transpose_b={operation.transpose_b:d}, "
            f"alpha={operation.alpha:f}, "
            f"beta={operation.beta:f}"
            ")"
        )

    def visit_GlobalAveragePool(self, operation: operations.GlobalAveragePool) -> None:
        self.generic_visit(operation)
        self.print_op_id(operation)
        print(f"GlobalAveragePool({self.get_op_id(operation.x)})")

    def visit_Identity(self, operation: operations.Identity) -> None:
        self.generic_visit(operation)
        self.print_op_id(operation)
        print(f"Identity({self.get_op_id(operation.x)})")

    def visit_Input(self, operation: operations.Input) -> None:
        self.generic_visit(operation)
        self.print_op_id(operation)
        print(f"Input({operation.shape}, dtype={operation.dtype})")

    def visit_LeakyRelu(self, operation: operations.LeakyRelu) -> None:
        self.generic_visit(operation)
        self.print_op_id(operation)
        print(f"LeakyRelu({self.get_op_id(operation.x)}, alpha={operation.alpha:f})")

    def visit_LogSoftmax(self, operation: operations.LogSoftmax) -> None:
        self.generic_visit(operation)
        self.print_op_id(operation)
        print(f"LogSoftmax({self.get_op_id(operation.x)}, axis={operation.axis})")

    def visit_MatMul(self, operation: operations.MatMul) -> None:
        self.generic_visit(operation)
        self.print_op_id(operation)
        print(f"MatMul({self.get_op_id(operation.a)}, {self.get_op_id(operation.b)})")

    def visit_MaxPool(self, operation: operations.MaxPool) -> None:
        self.generic_visit(operation)
        self.print_op_id(operation)
        print(f"MaxPool({self.get_op_id(operation.x)})")

    def visit_Mul(self, operation: operations.Mul) -> None:
        self.generic_visit(operation)
        self.print_op_id(operation)
        print(f"Mul({self.get_op_id(operation.a)}, {self.get_op_id(operation.b)})")

    def visit_OutputSelect(self, operation: operations.OutputSelect) -> None:
        self.generic_visit(operation)
        self.print_op_id(operation)
        print(f"OutputSelect({self.get_op_id(operation.operation)}, {operation.index})")

    def visit_Pad(self, operation: operations.Pad) -> None:
        self.generic_visit(operation)
        self.print_op_id(operation)
        print(f"Pad({self.get_op_id(operation.x)}, pads={operation.pads})")

    def visit_Relu(self, operation: operations.Relu) -> None:
        self.generic_visit(operation)
        self.print_op_id(operation)
        print(f"Relu({self.get_op_id(operation.x)})")

    def visit_Reshape(self, operation: operations.Reshape) -> None:
        self.generic_visit(operation)
        self.print_op_id(operation)
        print(
            f"Reshape({self.get_op_id(operation.x)}, {self.get_op_id(operation.shape)})"
        )

    def visit_Resize(self, operation: operations.Resize) -> None:
        self.generic_visit(operation)
        self.print_op_id(operation)
        inputs = []
        if operation.roi.size > 0:
            inputs.append(f"roi={operation.roi.tolist()}")
        if operation.scales.size > 0:
            inputs.append(f"scales={operation.scales.tolist()}")
        if operation.sizes.size > 0:
            inputs.append(f"sizes={operation.sizes.tolist()}")
        inputs_str = ", ".join(inputs)
        coord_transform_mode = operation.coordinate_transformation_mode
        print(
            (
                "Resize("
                f"{self.get_op_id(operation.x)}, "
                f"{inputs_str}, "
                f"coordinate_transformation_mode={coord_transform_mode}, "
                f"cubic_coeff_a={operation.cubic_coeff_a:f}, "
                f"exclude_outside={operation.exclude_outside:d}, "
                f"extrapolation_value={operation.extrapolation_value:f}, "
                f"mode={operation.mode}, "
                f"nearest_mode={operation.nearest_mode}"
                ")"
            )
        )

    def visit_Shape(self, operation: operations.Shape) -> None:
        self.generic_visit(operation)
        self.print_op_id(operation)
        print(f"Shape({self.get_op_id(operation.x)})")

    def visit_Sigmoid(self, operation: operations.Sigmoid) -> None:
        self.generic_visit(operation)
        self.print_op_id(operation)
        print(f"Sigmoid({self.get_op_id(operation.x)})")

    def visit_Slice(self, operation: operations.Slice) -> None:
        self.generic_visit(operation)
        self.print_op_id(operation)
        axes = (
            f", axes={self.get_op_id(operation.axes)}"
            if operation.axes is not None
            else ""
        )
        steps = (
            f", steps={self.get_op_id(operation.steps)}"
            if operation.steps is not None
            else ""
        )
        print(
            "Slice("
            f"{self.get_op_id(operation.x)}, "
            f"{self.get_op_id(operation.starts)}, "
            f"{self.get_op_id(operation.ends)}"
            f"{axes}{steps}"
            ")"
        )

    def visit_Sign(self, operation: operations.Sign) -> None:
        self.generic_visit(operation)
        self.print_op_id(operation)
        print(f"Sign({self.get_op_id(operation.x)})")

    def visit_Softmax(self, operation: operations.Softmax) -> None:
        self.generic_visit(operation)
        self.print_op_id(operation)
        print(f"Softmax({self.get_op_id(operation.x)}, axis={operation.axis})")

    def visit_Split(self, operation: operations.Sub) -> None:
        self.generic_visit(operation)
        self.print_op_id(operation)
        print(
            "Split(%s, axis=%s, split=%s)"
            % (self.get_op_id(operation.x), operation.axis, operation.split)
        )

    def visit_Sub(self, operation: operations.Sub) -> None:
        self.generic_visit(operation)
        self.print_op_id(operation)
        print(f"Sub({self.get_op_id(operation.a)}, {self.get_op_id(operation.b)})")

    def visit_Tanh(self, operation: operations.Tanh) -> None:
        self.generic_visit(operation)
        self.print_op_id(operation)
        print(f"Tanh({self.get_op_id(operation.x)})")

    def visit_Tile(self, operation: operations.Tile) -> None:
        self.generic_visit(operation)
        self.print_op_id(operation)
        print(f"Tile({self.get_op_id(operation.x)}, {operation.repeats})")

    def visit_Transpose(self, operation: operations.Transpose) -> None:
        self.generic_visit(operation)
        self.print_op_id(operation)
        print(
            f"Transpose({self.get_op_id(operation.x)},"
            f" permutation={operation.permutation})"
        )

    def visit_Unsqueeze(self, operation: operations.Unsqueeze) -> None:
        self.generic_visit(operation)
        self.print_op_id(operation)
        print(f"Unsqueeze({self.get_op_id(operation.x)}, axes={operation.axes})")


__all__ = [
    "OperationVisitor",
    "GetInputDetails",
    "OperationCounter",
    "EnsureSupportVisitor",
    "PrintVisitor",
]
