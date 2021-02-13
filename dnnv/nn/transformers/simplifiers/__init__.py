import numpy as np

from copy import copy
from typing import Any, Dict, List, Optional, Set, Type, Union

from ... import operations
from ...analyzers import Analysis, SplitAnalysis
from ...graph import OperationGraph
from ...operations import Operation
from ....utils import get_subclasses

from .base import ComposeSimplifiers, Simplifier
from .bundle_padding import BundlePadding
from .convert_batch_norm import ConvertBatchNorm
from .convert_matmul_add_to_gemm import ConvertMatMulAddToGemm
from .convert_reshape_to_flatten import ConvertReshapeToFlatten
from .drop_identities import DropIdentity, DropUnnecessaryConcat, DropUnnecessaryFlatten
from .move_activations_back import MoveActivationsBackward
from .squeeze_convs import SqueezeConvs
from .squeeze_gemms import SqueezeGemms

# @dlshriver: what was the use case here?
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


# @dlshriver: redundant with squeezegemms and matmuladdtogemm?
# class SqueezeMatMulAdds(Simplifier):
#     def visit_Add(self, operation: operations.Add):
#         if (
#             isinstance(operation.a, operations.MatMul)
#             and isinstance(operation.a.a, operations.Add)
#             and isinstance(operation.a.a.a, operations.MatMul)
#         ):
#             first_mm = operation.a.a.a
#             second_mm = operation.a
#             first_add = operation.a.a
#             second_add = operation

#             a = first_mm.a
#             b_0 = first_mm.b
#             b_1 = second_mm.b
#             b = np.matmul(b_0, b_1)
#             c = np.matmul(first_add.b, b_1) + second_add.b
#             return operations.Add(operations.MatMul(a, b), c)
#         # TODO : can also reduce in other cases, e.g., (MatMul >> MatMul >> Add)
#         return operation


def simplify(
    dnn: OperationGraph, simplifier: Optional[Simplifier] = None
) -> OperationGraph:
    if simplifier is None:
        simplifier = ComposeSimplifiers(dnn, *[c for c in get_subclasses(Simplifier)])
    simplified_graph = OperationGraph(dnn.walk(simplifier))
    return simplified_graph


__all__ = ["simplify"]
