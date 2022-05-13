from typing import Optional

from ....utils import get_subclasses
from ... import operations
from ...graph import OperationGraph
from ...operations import Operation
from .base import ComposeSimplifiers, Simplifier
from .bundle_padding import BundlePadding
from .bundle_transpose import BundleTranspose
from .convert_add import *
from .convert_batch_norm import *
from .convert_div_to_mul import *
from .convert_matmul_to_gemm import *
from .convert_mul import *
from .convert_reshape_to_flatten import *
from .convert_sub_to_add import *
from .drop_identities import *
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


def simplify(
    dnn: OperationGraph, simplifier: Optional[Simplifier] = None
) -> OperationGraph:
    if simplifier is None:
        simplifier = ComposeSimplifiers(
            *[c for c in get_subclasses(Simplifier) if not c is ComposeSimplifiers]
        )
    simplified_graph = simplifier.transform(dnn)
    simplified_graph._input_details = dnn._input_details
    simplified_graph._output_details = dnn._output_details
    return simplified_graph


__all__ = ["simplify"]
