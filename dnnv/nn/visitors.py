import numpy as np

from collections import namedtuple
from typing import List, Type

from . import operations
from .operations import Operation
from .utils import TensorDetails


class OperationVisitor:
    def visit(self, operation: Operation):
        method_name = "visit_%s" % operation.__class__.__name__
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
        if operation not in self.visited:
            super().visit(operation)
            self.visited.add(operation)
        return tuple(self.input_details)

    def visit_Input(self, operation: operations.Input):
        self.input_details.append(
            TensorDetails(tuple(int(i) for i in operation.shape), operation.dtype)
        )


class OperationCounter(OperationVisitor):
    def __init__(self):
        self.visited = set()

    def visit(self, operation: Operation):
        if operation not in self.visited:
            self.visited.add(operation)
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
        return super().visit(operation)

    def generic_visit(self, operation: Operation):
        if not hasattr(self, "visit_%s" % operation.__class__.__name__):
            raise ValueError(operation)
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
        return "%s_%s" % (op_type, idx)

    def print_op_id(self, operation: Operation) -> None:
        op_id = self.get_op_id(operation)
        print("%-32s" % op_id, end=": ")

    def visit_Add(self, operation: operations.Add) -> None:
        self.generic_visit(operation)
        self.print_op_id(operation)
        print(
            "Add(%s, %s)" % (self.get_op_id(operation.a), self.get_op_id(operation.b))
        )

    def visit_Atan(self, operation: operations.Atan) -> None:
        self.generic_visit(operation)
        self.print_op_id(operation)
        print("Atan(%s)" % self.get_op_id(operation.x))

    def visit_AveragePool(self, operation: operations.AveragePool) -> None:
        self.generic_visit(operation)
        self.print_op_id(operation)
        print(
            "AveragePool(%s, %s)"
            % (self.get_op_id(operation.x), operation.kernel_shape)
        )

    def visit_BatchNormalization(
        self, operation: operations.BatchNormalization
    ) -> None:
        self.generic_visit(operation)
        self.print_op_id(operation)
        print("BatchNormalization(%s)" % self.get_op_id(operation.x))

    def visit_Concat(self, operation: operations.Concat) -> None:
        self.generic_visit(operation)
        self.print_op_id(operation)
        print("Concat(%s)" % ([self.get_op_id(x) for x in operation.x],))

    def visit_Conv(self, operation: operations.Conv) -> None:
        self.generic_visit(operation)
        self.print_op_id(operation)
        print(
            "Conv(%s, kernel_shape=%s, strides=%s, pads=%s)"
            % (
                self.get_op_id(operation.x),
                operation.kernel_shape,
                operation.strides,
                operation.pads,
            )
        )

    def visit_Dropout(self, operation: operations.Dropout) -> None:
        self.generic_visit(operation)
        self.print_op_id(operation)
        print("Dropout(%s, ratio=%s)" % (self.get_op_id(operation.x), operation.ratio))

    def visit_Elu(self, operation: operations.Elu) -> None:
        self.generic_visit(operation)
        self.print_op_id(operation)
        print("Elu(%s)" % self.get_op_id(operation.x))

    def visit_Flatten(self, operation: operations.Flatten) -> None:
        self.generic_visit(operation)
        self.print_op_id(operation)
        print("Flatten(%s, axis=%s)" % (self.get_op_id(operation.x), operation.axis))

    def visit_Gather(self, operation: operations.Gather) -> None:
        self.generic_visit(operation)
        self.print_op_id(operation)
        print(
            "Gather(%s, %s, axis=%s)"
            % (
                self.get_op_id(operation.x),
                self.get_op_id(operation.indices),
                operation.axis,
            )
        )

    def visit_Gemm(self, operation: operations.Gemm) -> None:
        self.generic_visit(operation)
        self.print_op_id(operation)
        print(
            "Gemm(%s, %s, %s)"
            % (
                self.get_op_id(operation.a),
                self.get_op_id(operation.b),
                self.get_op_id(operation.c),
            )
        )

    def visit_GlobalAveragePool(self, operation: operations.GlobalAveragePool) -> None:
        self.generic_visit(operation)
        self.print_op_id(operation)
        print("GlobalAveragePool(%s)" % self.get_op_id(operation.x))

    def visit_Identity(self, operation: operations.Identity) -> None:
        self.generic_visit(operation)
        self.print_op_id(operation)
        print("Identity(%s)" % self.get_op_id(operation.x))

    def visit_Input(self, operation: operations.Input) -> None:
        self.generic_visit(operation)
        self.print_op_id(operation)
        print("Input(%s, dtype=%s)" % (operation.shape, operation.dtype.name))

    def visit_LogSoftmax(self, operation: operations.LogSoftmax) -> None:
        self.generic_visit(operation)
        self.print_op_id(operation)
        print("LogSoftmax(%s, axis=%s)" % (self.get_op_id(operation.x), operation.axis))

    def visit_MatMul(self, operation: operations.MatMul) -> None:
        self.generic_visit(operation)
        self.print_op_id(operation)
        print(
            "MatMul(%s, %s)"
            % (self.get_op_id(operation.a), self.get_op_id(operation.b))
        )

    def visit_MaxPool(self, operation: operations.MaxPool) -> None:
        self.generic_visit(operation)
        self.print_op_id(operation)
        print("MaxPool(%s)" % self.get_op_id(operation.x))

    def visit_Mul(self, operation: operations.Mul) -> None:
        self.generic_visit(operation)
        self.print_op_id(operation)
        print(
            "Mul(%s, %s)" % (self.get_op_id(operation.a), self.get_op_id(operation.b))
        )

    def visit_OutputSelect(self, operation: operations.OutputSelect) -> None:
        self.generic_visit(operation)
        self.print_op_id(operation)
        print(
            "OutputSelect(%s, %s)"
            % (self.get_op_id(operation.operation), operation.index)
        )

    def visit_Pad(self, operation: operations.Pad) -> None:
        self.generic_visit(operation)
        self.print_op_id(operation)
        print("Pad(%s, pads=%s)" % (self.get_op_id(operation.x), operation.pads))

    def visit_Relu(self, operation: operations.Relu) -> None:
        self.generic_visit(operation)
        self.print_op_id(operation)
        print("Relu(%s)" % self.get_op_id(operation.x))

    def visit_Reshape(self, operation: operations.Reshape) -> None:
        self.generic_visit(operation)
        self.print_op_id(operation)
        print(
            "Reshape(%s, %s)"
            % (self.get_op_id(operation.x), self.get_op_id(operation.shape))
        )

    def visit_Shape(self, operation: operations.Shape) -> None:
        self.generic_visit(operation)
        self.print_op_id(operation)
        print("Shape(%s)" % self.get_op_id(operation.x))

    def visit_Sigmoid(self, operation: operations.Sigmoid) -> None:
        self.generic_visit(operation)
        self.print_op_id(operation)
        print("Sigmoid(%s)" % self.get_op_id(operation.x))

    def visit_Softmax(self, operation: operations.Softmax) -> None:
        self.generic_visit(operation)
        self.print_op_id(operation)
        print("Softmax(%s, axis=%s)" % (self.get_op_id(operation.x), operation.axis))

    def visit_Tanh(self, operation: operations.Tanh) -> None:
        self.generic_visit(operation)
        self.print_op_id(operation)
        print("Tanh(%s)" % self.get_op_id(operation.x))

    def visit_Transpose(self, operation: operations.Transpose) -> None:
        self.generic_visit(operation)
        self.print_op_id(operation)
        print(
            "Transpose(%s, permutation=%s)"
            % (self.get_op_id(operation.x), operation.permutation)
        )

    def visit_Unsqueeze(self, operation: operations.Unsqueeze) -> None:
        self.generic_visit(operation)
        self.print_op_id(operation)
        print("Unsqueeze(%s, axes=%s)" % (self.get_op_id(operation.x), operation.axes))
