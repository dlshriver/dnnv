import numpy as np

from collections import namedtuple

from .operations import Operation


class OperationVisitor:
    def visit(self, operation):
        method_name = "visit_%s" % operation.__class__.__name__
        visitor = getattr(self, method_name, self.generic_visit)
        return visitor(operation)

    def generic_visit(self, operation):
        for value in operation.__dict__.values():
            if isinstance(value, Operation):
                self.visit(value)
            elif isinstance(value, (list, tuple)):
                for sub_value in value:
                    if isinstance(sub_value, Operation):
                        self.visit(sub_value)
        return operation


class GetInputDetails(OperationVisitor):
    InputDetails = namedtuple("InputDetails", ["shape", "dtype"])

    def __init__(self):
        self.visited = set()
        self.input_details = []

    def visit(self, operation):
        if operation not in self.visited:
            super().visit(operation)
            self.visited.add(operation)
        return tuple(self.input_details)

    def visit_Input(self, operation):
        self.input_details.append(
            self.InputDetails(
                tuple((i if i > 0 else 1) for i in operation.shape), operation.dtype
            )
        )


class OperationCounter(OperationVisitor):
    def __init__(self):
        self.visited = set()

    def visit(self, operation):
        if operation not in self.visited:
            self.visited.add(operation)
            super().generic_visit(operation)
        return len(self.visited)


class PrintVisitor(OperationVisitor):
    def __init__(self):
        super().__init__()
        self.visited = set()
        self.identifiers = {"count": {}, "op": {}}

    def visit(self, operation):
        if operation in self.visited:
            return
        self.visited.add(operation)
        return super().visit(operation)

    def generic_visit(self, operation):
        if not hasattr(self, "visit_%s" % operation.__class__.__name__):
            raise ValueError(operation)
        super().generic_visit(operation)

    def get_op_id(self, operation):
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

    def print_op_id(self, operation):
        op_id = self.get_op_id(operation)
        print("%-32s" % op_id, end=": ")

    def visit_Add(self, operation):
        self.generic_visit(operation)
        self.print_op_id(operation)
        print(
            "Add(%s, %s)" % (self.get_op_id(operation.a), self.get_op_id(operation.b))
        )

    def visit_Atan(self, operation):
        self.generic_visit(operation)
        self.print_op_id(operation)
        print("Atan(%s)" % self.get_op_id(operation.x))

    def visit_AveragePool(self, operation):
        self.generic_visit(operation)
        self.print_op_id(operation)
        print(
            "AveragePool(%s, %s)"
            % (self.get_op_id(operation.x), operation.kernel_shape)
        )

    def visit_BatchNormalization(self, operation):
        self.generic_visit(operation)
        self.print_op_id(operation)
        print("BatchNormalization(%s)" % self.get_op_id(operation.x))

    def visit_Concat(self, operation):
        self.generic_visit(operation)
        self.print_op_id(operation)
        print("Concat(%s)" % ([self.get_op_id(x) for x in operation.x],))

    def visit_Conv(self, operation):
        self.generic_visit(operation)
        self.print_op_id(operation)
        print("Conv(%s)" % self.get_op_id(operation.x))

    def visit_Dropout(self, operation):
        self.generic_visit(operation)
        self.print_op_id(operation)
        print("Dropout(%s, ratio=%s)" % (self.get_op_id(operation.x), operation.ratio))

    def visit_Flatten(self, operation):
        self.generic_visit(operation)
        self.print_op_id(operation)
        print("Flatten(%s, axis=%s)" % (self.get_op_id(operation.x), operation.axis))

    def visit_Gather(self, operation):
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

    def visit_Gemm(self, operation):
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

    def visit_GlobalAveragePool(self, operation):
        self.generic_visit(operation)
        self.print_op_id(operation)
        print("GlobalAveragePool(%s)" % self.get_op_id(operation.x))

    def visit_Identity(self, operation):
        self.generic_visit(operation)
        self.print_op_id(operation)
        print("Identity(%s)" % self.get_op_id(operation.x))

    def visit_Input(self, operation):
        self.generic_visit(operation)
        self.print_op_id(operation)
        print("Input(%s, dtype=%s)" % (operation.shape, operation.dtype.name))

    def visit_LogSoftmax(self, operation):
        self.generic_visit(operation)
        self.print_op_id(operation)
        print("LogSoftmax(%s, axis=%s)" % (self.get_op_id(operation.x), operation.axis))

    def visit_MatMul(self, operation):
        self.generic_visit(operation)
        self.print_op_id(operation)
        print(
            "MatMul(%s, %s)"
            % (self.get_op_id(operation.a), self.get_op_id(operation.b))
        )

    def visit_MaxPool(self, operation):
        self.generic_visit(operation)
        self.print_op_id(operation)
        print("MaxPool(%s)" % self.get_op_id(operation.x))

    def visit_Mul(self, operation):
        self.generic_visit(operation)
        self.print_op_id(operation)
        print(
            "Mul(%s, %s)" % (self.get_op_id(operation.a), self.get_op_id(operation.b))
        )

    def visit_OutputSelect(self, operation):
        self.generic_visit(operation)
        self.print_op_id(operation)
        print(
            "OutputSelect(%s, %s)"
            % (self.get_op_id(operation.operation), operation.index)
        )

    def visit_Pad(self, operation):
        self.generic_visit(operation)
        self.print_op_id(operation)
        print("Pad(%s, pads=%s)" % (self.get_op_id(operation.x), operation.pads))

    def visit_Relu(self, operation):
        self.generic_visit(operation)
        self.print_op_id(operation)
        print("Relu(%s)" % self.get_op_id(operation.x))

    def visit_Reshape(self, operation):
        self.generic_visit(operation)
        self.print_op_id(operation)
        print(
            "Reshape(%s, %s)"
            % (self.get_op_id(operation.x), self.get_op_id(operation.shape))
        )

    def visit_Shape(self, operation):
        self.generic_visit(operation)
        self.print_op_id(operation)
        print("Shape(%s)" % self.get_op_id(operation.x))

    def visit_Sigmoid(self, operation):
        self.generic_visit(operation)
        self.print_op_id(operation)
        print("Sigmoid(%s)" % self.get_op_id(operation.x))

    def visit_Softmax(self, operation):
        self.generic_visit(operation)
        self.print_op_id(operation)
        print("Softmax(%s, axis=%s)" % (self.get_op_id(operation.x), operation.axis))

    def visit_Tanh(self, operation):
        self.generic_visit(operation)
        self.print_op_id(operation)
        print("Tanh(%s)" % self.get_op_id(operation.x))

    def visit_Transpose(self, operation):
        self.generic_visit(operation)
        self.print_op_id(operation)
        print(
            "Transpose(%s, permutation=%s)"
            % (self.get_op_id(operation.x), operation.permutation)
        )

    def visit_Unsqueeze(self, operation):
        self.generic_visit(operation)
        self.print_op_id(operation)
        print("Unsqueeze(%s, axes=%s)" % (self.get_op_id(operation.x), operation.axes))
