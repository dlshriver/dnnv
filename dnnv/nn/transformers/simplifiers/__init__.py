from typing import Optional

from ....utils import get_subclasses
from ... import operations
from ...graph import OperationGraph
from ...operations import Operation
from .base import *
from .bundle_padding import *
from .bundle_transpose import *
from .convert_add import *
from .convert_batch_norm import *
from .convert_div_to_mul import *
from .convert_matmul_to_gemm import *
from .convert_mul import *
from .convert_reshape_to_flatten import *
from .convert_sub_to_add import *
from .drop_identities import *
from .move_activations_back import *
from .squeeze_convs import *
from .squeeze_gemms import *


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


def simplify(
    dnn: OperationGraph, simplifier: Optional[Simplifier] = None
) -> OperationGraph:
    if simplifier is None:
        simplifier = ComposeSimplifiers(
            dnn, *[c for c in get_subclasses(Simplifier) if not c is ComposeSimplifiers]
        )
    simplified_graph = OperationGraph(dnn.walk(simplifier))
    return simplified_graph


__all__ = ["simplify"]
