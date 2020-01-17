import numpy as np

from copy import deepcopy

from . import operations
from .graph import OperationGraph
from .operations import Operation, Input
from .visitors import OperationVisitor, GetInputDetails


class OperationTransformer(OperationVisitor):
    def generic_visit(self, operation):
        for name, value in operation.__dict__.items():
            if isinstance(value, Operation):
                new_value = self.visit(value)
                setattr(operation, name, new_value)
            elif isinstance(value, (list, set, tuple)):
                new_value = []
                for value_ in value:
                    if isinstance(value_, Operation):
                        new_value_ = self.visit(value_)
                        new_value.append(new_value_)
                    else:
                        new_value.append(value_)
                setattr(operation, name, type(value)(new_value))
        return operation


class DropPrefix(OperationTransformer):
    def __init__(self, prefix_graph):
        self.prefix_graph = prefix_graph
        self._cache = {}

    def visit(self, operation):
        if operation not in self._cache:
            if operation not in self.prefix_graph.output_operations:
                result = super().visit(operation)
            else:
                input_details = GetInputDetails().visit(operation)
                y = OperationGraph([operation])(
                    *[
                        np.ones([i if i >= 0 else 1 for i in d[0]], dtype=d[1])
                        for d in input_details
                    ],
                    squeeze=False,
                )
                if len(y) > 1:
                    raise ValueError(
                        "Dropping prefixes with multiple output values is currently not supported"
                    )
                new_input_shape = np.asarray(y[0].shape)
                if all(d[0][0] < 0 for d in input_details) and new_input_shape[0] == 1:
                    new_input_shape[0] = -1
                result = Input(new_input_shape, y[0].dtype)
            self._cache[operation] = result
        return self._cache[operation]

    def generic_visit(self, operation):
        kwargs = {}
        for name, value in operation.__dict__.items():
            if isinstance(value, Operation):
                new_value = self.visit(value)
                kwargs[name] = new_value
            else:
                kwargs[name] = deepcopy(value)
        return operation.__class__(**kwargs)


class Simplify(OperationTransformer):
    def __init__(self):
        self._cache = {}

    def visit(self, operation):
        if operation not in self._cache:
            result = super().visit(operation)
            self._cache[operation] = result
        return self._cache[operation]

    def visit_Add(self, operation: operations.Add):
        operation = super().generic_visit(operation)
        if isinstance(operation, operations.Add):
            if isinstance(operation.a, Operation):
                input_op = operation.a
                c = operation.b
            else:
                input_op = operation.b
                c = operation.a
            if isinstance(input_op, operations.MatMul):
                a = input_op.a
                b = input_op.b
                return self.visit(operations.Gemm(a, b, c))
        return operation

    def visit_BatchNormalization(self, operation: operations.BatchNormalization):
        operation = super().generic_visit(operation)
        input_op = operation.x
        if isinstance(input_op, operations.Conv):
            std = np.sqrt(operation.variance + operation.epsilon)
            a = operation.scale / std
            b = operation.bias - operation.scale * operation.mean / std

            weights = input_op.w
            a_w = a[:, None, None, None]
            weights = a_w * weights
            bias = input_op.b
            if bias is None:
                bias = np.zeros(weights.shape[0])
            bias = a * bias + b

            input_op.w = weights
            input_op.b = bias
            return input_op
        elif isinstance(input_op, operations.Input):
            c = operation.mean.shape[0]
            std = np.sqrt(operation.variance + operation.epsilon)
            k = np.zeros((c, c, 1, 1))  # identity kernel (H, W, inC, outC)
            for i in range(c):
                k[i, i, 0, 0] = 1
            W = k * operation.scale / std
            b = operation.bias - operation.scale * operation.mean / std
            op = operations.Conv(input_op, W, b)
            return op
        return operation

    def visit_Concat(self, operation: operations.Concat):
        operation = super().generic_visit(operation)
        if all(not isinstance(x, Operation) for x in operation.x):
            return np.concatenate([x for x in operation.x])
        return operation

    def visit_Conv(self, operation: operations.Conv):
        operation = super().generic_visit(operation)
        input_op = operation.x
        if not isinstance(input_op, operations.Pad):
            return operation
        pads = operation.pads
        if input_op.mode != "constant" or input_op.value != 0.0:
            return operation
        if not np.all(p == 0 for p in input_op.pads[:4]):
            return operation
        pad_top, pad_left = input_op.pads[2:4]
        pad_bottom, pad_right = input_op.pads[6:8]
        operation.pads = pads + np.array([pad_top, pad_left, pad_bottom, pad_right])
        operation.x = input_op.x
        return operation

    def visit_Gather(self, operation: operations.Gather):
        operation = super().generic_visit(operation)
        if not isinstance(operation.x, Operation) and not isinstance(
            operation.indices, Operation
        ):
            return np.take(operation.x, operation.indices, axis=operation.axis)
        return operation

    def visit_Gemm(self, operation: operations.Gemm):
        operation = super().generic_visit(operation)
        if isinstance(operation.a, operations.Gemm) and not operation.transpose_a:
            input_op = operation.a
            if (
                not isinstance(input_op.a, Operation)
                or input_op.alpha != 1.0
                or input_op.beta != 1.0
            ):
                return operation
            a = input_op.a
            b_0 = input_op.b.T if input_op.transpose_b else input_op.b
            b_1 = operation.b.T if operation.transpose_b else operation.b
            b = np.matmul(b_0, b_1)
            c = np.matmul(input_op.c, b_1) + operation.c
            return operations.Gemm(
                a,
                b,
                c,
                transpose_a=input_op.transpose_a,
                alpha=operation.alpha,
                beta=operation.beta,
            )
        # TODO : reduce when operation.b is Gemm
        return operation

    def visit_Identity(self, operation: operations.Identity):
        operation = super().generic_visit(operation)
        return operation.x

    def visit_Relu(self, operation: operations.Relu):
        operation = super().generic_visit(operation)
        input_op = operation.x
        if not isinstance(
            input_op, (operations.Reshape, operations.Transpose, operations.Flatten)
        ):
            return operation
        input_ops = [input_op]
        while isinstance(
            input_op, (operations.Reshape, operations.Transpose, operations.Flatten)
        ):
            input_op = input_ops[-1].x
            input_ops.append(input_op)
        output_op = operation.x
        operation.x = input_op
        input_ops[-2].x = operation
        return output_op

    def visit_Shape(self, operation: operations.Shape):
        operation = super().generic_visit(operation)
        input_op = operation.x
        return OperationGraph([input_op]).output_shape[0]

    def visit_Unsqueeze(self, operation: operations.Unsqueeze):
        operation = super().generic_visit(operation)
        if not isinstance(operation.x, Operation):
            x = operation.x
            for axis in operation.axes:
                x = np.expand_dims(x, axis)
            return x
        return operation


class Slicer(OperationTransformer):
    def __init__(self, start, stop):
        self.start = start
        self.stop = stop

        self.index = 0
        self.length = 0
        self._index_cache = {}
        self.current_pass = None
        self._cache = {}

    def visit(self, operation):
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

    def generic_visit(self, operation):
        if self.current_pass == "indexing":
            self.index -= 1
            self.length = max(self.length, -self.index)
            if operation not in self._index_cache:
                self._index_cache[operation] = [float("inf"), float("-inf"), operation]
            pos_index, neg_index, _ = self._index_cache[operation]
            self._index_cache[operation][1] = max(neg_index, self.index)
            result = super().generic_visit(operation)
            pos_index, neg_index, _ = self._index_cache[operation]
            self._index_cache[operation][0] = min(pos_index, self.length + self.index)
            self.index += 1
            return result
        elif self.current_pass == "selection":
            pos_index, neg_index, _ = self._index_cache[operation]
            if (self.start > 0 and pos_index < self.start) or (
                self.start < 0 and neg_index < self.start
            ):
                input_details = GetInputDetails().visit(operation)
                y = OperationGraph([operation])(
                    *[
                        np.ones([i if i >= 0 else 1 for i in d[0]], dtype=d[1])
                        for d in input_details
                    ],
                    squeeze=False,
                )
                return Input(y[0].shape, y[0].dtype)
            kwargs = {}
            for name, value in operation.__dict__.items():
                if isinstance(value, Operation):
                    new_value = self.visit(value)
                    kwargs[name] = new_value
                elif isinstance(value, (tuple, list, set)):
                    new_value = []
                    for value_ in value:
                        if isinstance(value_, Operation):
                            new_value_ = self.visit(value_)
                            new_value.append(new_value_)
                        else:
                            new_value.append(deepcopy(value_))
                    kwargs[name] = type(value)(new_value)
                else:
                    kwargs[name] = deepcopy(value)
            return operation.__class__(**kwargs)
        else:
            raise ValueError(f"Unknown slicing pass: {self.current_pass}")
