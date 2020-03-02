import numpy as np

from typing import Optional, Union

from .. import operations
from ..graph import OperationGraph
from ..operations import Operation
from ...utils import get_subclasses

from .base import OperationTransformer


class Simplifier(OperationTransformer):
    def __init__(self):
        self._cache = {}
        self._modified_graph = False

    def visit(self, operation: Operation) -> Operation:
        if operation not in self._cache:
            operation = super().generic_visit(operation)
            result = super().visit(operation)
            if result is not operation:
                self._modified_graph = True
            self._cache[operation] = result
        return self._cache[operation]


class Compose(Simplifier):
    def __init__(self, *simplifiers: Simplifier):
        super().__init__()
        self.simplifiers = simplifiers

    def visit(self, operation: Operation) -> Operation:
        modified_graph = True
        while modified_graph:
            modified_graph = False
            for simplifier in self.simplifiers:
                simplifier._modified_graph = False
                operation = simplifier.visit(operation)
                modified_graph |= simplifier._modified_graph
            self._modified_graph |= modified_graph
        return operation


class BundleConvPadding(Simplifier):
    def visit_Conv(self, operation: operations.Conv) -> operations.Conv:
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


class ConvertBatchNorm(Simplifier):
    def visit_BatchNormalization(self, operation: operations.BatchNormalization):
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
        # TODO : in what other scenarios can BatchNorm be converted?
        return operation


class DropIdentity(Simplifier):
    def visit_Identity(self, operation: operations.Identity):
        return operation.x


class MatMulAddToGemm(Simplifier):
    def visit_Add(
        self, operation: operations.Add
    ) -> Union[operations.Add, operations.Gemm]:
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
                return operations.Gemm(a, b, c)
        return operation


class MoveActivationsBackward(Simplifier):
    ShapingOperation = operations.Reshape | operations.Transpose | operations.Flatten

    def move_back(self, operation: Union[operations.Relu, operations.Sigmoid]):
        if next(self.ShapingOperation.match([operation.x]), None) is None:
            return operation
        output_op = operation.x
        input_op = operation.x.x
        next_op = operation
        while next(self.ShapingOperation.match([input_op]), None) is not None:
            next_op = input_op
            input_op = input_op.x
        operation.x = input_op
        next_op.x = operation
        return output_op

    def visit_Relu(self, operation: operations.Relu):
        return self.move_back(operation)

    def visit_Sigmoid(self, operation: operations.Sigmoid):
        return self.move_back(operation)


class PropagateConstants(Simplifier):
    def visit_Concat(
        self, operation: operations.Concat
    ) -> Union[np.ndarray, operations.Concat]:
        if all(not isinstance(x, Operation) for x in operation.x):
            return np.concatenate([x for x in operation.x])
        return operation

    def visit_Gather(
        self, operation: operations.Gather
    ) -> Union[np.ndarray, operations.Gather]:
        if not isinstance(operation.x, Operation) and not isinstance(
            operation.indices, Operation
        ):
            return np.take(operation.x, operation.indices, axis=operation.axis)
        return operation

    def visit_Shape(
        self, operation: operations.Shape
    ) -> Union[np.ndarray, operations.Shape]:
        input_op = operation.x
        return OperationGraph([input_op]).output_shape[0]

    def visit_Unsqueeze(
        self, operation: operations.Unsqueeze
    ) -> Union[np.ndarray, operations.Unsqueeze]:
        if not isinstance(operation.x, Operation):
            x = operation.x
            for axis in operation.axes:
                x = np.expand_dims(x, axis)
            return x
        return operation


class SqueezeGemms(Simplifier):
    def visit_Gemm(self, operation: operations.Gemm) -> operations.Gemm:
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


def simplify(
    dnn: OperationGraph, simplifier: Optional[Simplifier] = None
) -> OperationGraph:
    if simplifier is None:
        simplifier = Compose(*[c() for c in get_subclasses(Simplifier)])
    return OperationGraph(dnn.walk(simplifier))


__all__ = ["simplify"]
