import numpy as np

from copy import copy
from typing import Any, Dict, List, Optional, Set, Type, Union

from .. import operations
from ..graph import OperationGraph
from ..operations import Operation
from ...utils import get_subclasses

from .base import OperationTransformer, OperationVisitor


class Analysis(OperationVisitor):
    def __init__(self, dnn: OperationGraph):
        self.results: Dict[Operation, Any] = {}
        dnn.walk(self)

    def __getitem__(self, index):
        return self.results[index]


class SplitAnalysis(Analysis):
    def __init__(self, dnn: OperationGraph):
        self._cache: Set[Operation] = set()
        super().__init__(dnn)

    def visit(self, operation: Operation):
        if operation not in self._cache:
            self._cache.add(operation)
            super().visit(operation)
            self.results[operation] = False
        else:
            self.results[operation] = True
        return self


class Simplifier(OperationTransformer):
    ANALYSES: Dict[str, Type[Analysis]] = {}

    def __init__(self, dnn: OperationGraph):
        self._cache: Dict[Operation, Operation] = {}
        self._modified_graph = False
        self.dnn = dnn
        self.analysis: Dict[str, Analysis] = {
            name: analysis(dnn) for name, analysis in self.ANALYSES.items()
        }

    def visit(self, operation: Operation) -> Operation:
        if operation not in self._cache:
            operation = super().generic_visit(operation)
            result = super().visit(operation)
            if result is not operation:
                self._modified_graph = True
            self._cache[operation] = result
        return self._cache[operation]


class Compose(Simplifier):
    def __init__(self, dnn: OperationGraph, *simplifiers: Type[Simplifier]):
        super().__init__(dnn)
        self.simplifiers = [simplifier(dnn) for simplifier in simplifiers]

    def visit(self, operation: Operation) -> Operation:
        modified_graph = True
        while modified_graph:
            modified_graph = False
            for simplifier in self.simplifiers:
                simplifier._modified_graph = False
                simplifier._cache = {}
                operation = simplifier.visit(operation)
                modified_graph |= simplifier._modified_graph
            self._modified_graph |= modified_graph
        return operation


class BundlePadding(Simplifier):
    def visit_Conv(self, operation: operations.Conv) -> operations.Conv:
        input_op = operation.x
        if not isinstance(input_op, operations.Pad):
            return operation
        pads = operation.pads
        if input_op.mode != "constant" or input_op.value != 0.0:
            return operation
        if not np.all(p == 0 for p in input_op.pads[:4]):
            return operation
        operation = copy(operation)
        pad_top, pad_left = input_op.pads[2:4]
        pad_bottom, pad_right = input_op.pads[6:8]
        operation.pads = pads + np.array([pad_top, pad_left, pad_bottom, pad_right])
        operation.x = input_op.x
        return operation

    def visit_MaxPool(self, operation: operations.MaxPool) -> operations.MaxPool:
        input_op = operation.x
        if not isinstance(input_op, operations.Pad):
            return operation
        pads = operation.pads
        if input_op.mode != "constant" or input_op.value != 0.0:
            return operation
        if not np.all(p == 0 for p in input_op.pads[:4]):
            return operation
        operation = copy(operation)
        pad_top, pad_left = input_op.pads[2:4]
        pad_bottom, pad_right = input_op.pads[6:8]
        operation.pads = pads + np.array([pad_top, pad_left, pad_bottom, pad_right])
        operation.x = input_op.x
        return operation


class ConvertBatchNorm(Simplifier):
    ANALYSES = {"is_split": SplitAnalysis}

    def visit_BatchNormalization(self, operation: operations.BatchNormalization):
        input_op = operation.x
        if (
            isinstance(input_op, operations.Conv)
            and not self.analysis["is_split"][input_op]
        ):
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


class MatMulVectorArgsReorder(Simplifier):
    def visit_MatMul(self, operation: operations.MatMul) -> operations.MatMul:
        if (
            not isinstance(operation.a, Operation)
            and isinstance(operation.b, Operation)
            and len(OperationGraph([operation.b]).output_shape[0]) == 1
            and len(operation.a.shape) == 2
        ):
            return operations.MatMul(operation.b, operation.a.T)
        return operation


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
                if isinstance(a, Operation):
                    a_ndim = len(OperationGraph([a]).output_shape[0])
                else:
                    a_ndim = len(a.shape)
                if isinstance(b, Operation):
                    b_ndim = len(OperationGraph([b]).output_shape[0])
                else:
                    b_ndim = len(b.shape)
                if a_ndim == 2 and b_ndim == 2:
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


class ReshapeToFlatten(Simplifier):
    def visit_Reshape(self, operation: operations.Reshape):
        if not isinstance(operation.shape, operations.Concat):
            return operation
        concat: operations.Concat = operation.shape
        if concat.axis != 0:
            return operation
        if len(concat.x) != 2 or concat.x[1] != -1:
            return operation
        if not isinstance(concat.x[0], operations.Unsqueeze):
            return operation
        unsqueeze: operations.Unsqueeze = concat.x[0]
        if len(unsqueeze.axes) != 1 or unsqueeze.axes[0] != 0:
            return operation
        if not isinstance(unsqueeze.x, operations.Gather):
            return operation
        gather: operations.Gather = unsqueeze.x
        if gather.axis != 0 or gather.indices.shape != () or gather.indices != 0:
            return operation
        if not isinstance(gather.x, operations.Shape):
            return operation
        shape: operations.Shape = gather.x
        return operations.Flatten(shape.x, axis=1)


class SqueezeConvs(Simplifier):
    def is_diagonal(self, array):
        i, j = array.shape
        return ~np.any(array.reshape(-1)[:-1].reshape(i - 1, j + 1)[:, 1:])

    def visit_Conv(self, operation: operations.Conv) -> operations.Conv:
        if (
            isinstance(operation.x, operations.Conv)
            and operation.x.w.shape[2] == 1
            and operation.x.w.shape[3] == 1
            and all(s == 1 for s in operation.x.strides)
            and all(p == 0 for p in operation.x.pads)
            and all(d == 1 for d in operation.x.dilations)
            and operation.x.group == 1
            and self.is_diagonal(operation.x.w[:, :, 0, 0])
        ):
            w = np.diag(operation.x.w[:, :, 0, 0]).reshape((1, -1, 1, 1))
            b = operation.x.b

            out_c, in_c, k_h, k_w = operation.w.shape

            weights = operation.w * np.tile(w, (out_c, 1, k_h, k_w))
            bias = operation.b + (operation.w * np.tile(b, (out_c, 1, k_h, k_w))).sum(
                axis=(1, 2, 3)
            )

            op = copy(operation)
            op.x = operation.x.x
            op.w = weights
            op.b = bias
            return op
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


class SqueezeMatMulAdds(Simplifier):
    def visit_Add(self, operation: operations.Add):
        if (
            isinstance(operation.a, operations.MatMul)
            and isinstance(operation.a.a, operations.Add)
            and isinstance(operation.a.a.a, operations.MatMul)
        ):
            first_mm = operation.a.a.a
            second_mm = operation.a
            first_add = operation.a.a
            second_add = operation

            a = first_mm.a
            b_0 = first_mm.b
            b_1 = second_mm.b
            b = np.matmul(b_0, b_1)
            c = np.matmul(first_add.b, b_1) + second_add.b
            return operations.Add(operations.MatMul(a, b), c)
        # TODO : can also reduce in other cases, e.g., (MatMul >> MatMul >> Add)
        return operation


class DropUnnecessaryConcat(Simplifier):
    def visit_Concat(self, operation: operations.Concat) -> operations.Operation:
        if len(operation.x) == 1:
            return operation.x[0]
        return operation


class DropUnnecessaryFlatten(Simplifier):
    def visit_Flatten(self, operation: operations.Flatten) -> operations.Operation:
        if (
            operation.axis == 1
            and len(OperationGraph([operation.x]).output_shape[0]) == 2
        ):
            return operation.x
        return operation


def simplify(
    dnn: OperationGraph, simplifier: Optional[Simplifier] = None
) -> OperationGraph:
    if simplifier is None:
        simplifier = Compose(dnn, *[c for c in get_subclasses(Simplifier)])
    simplified_graph = OperationGraph(dnn.walk(simplifier))
    return simplified_graph


__all__ = ["simplify"]
