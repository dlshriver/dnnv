import logging
import os
import typing

from ....errors import DNNVError
from ...graph import OperationGraph
from .base import ComposeSimplifiers, Simplifier
from .bundle_padding import BundlePadding
from .bundle_transpose import BundleTranspose
from .convert_add import ConvertAdd
from .convert_batch_norm import ConvertBatchNorm
from .convert_div_to_mul import ConvertDivToMul
from .convert_matmul_to_gemm import ConvertMatMulToGemm
from .convert_mul import ConvertMul
from .convert_reshape_to_flatten import ConvertReshapeToFlatten
from .convert_sub_to_add import ConvertSubToAdd
from .drop_identities import (
    DropDropout,
    DropIdentity,
    DropUnnecessaryConcat,
    DropUnnecessaryFlatten,
    DropUnnecessaryRelu,
)
from .move_activations_back import MoveActivationsBackward
from .reluify_maxpool import ReluifyMaxPool
from .squeeze_convs import SqueezeConvs
from .squeeze_gemms import SqueezeGemms

DEFAULT_SIMPLIFIERS = [
    BundlePadding,
    BundleTranspose,
    ConvertAdd,
    ConvertBatchNorm,
    ConvertDivToMul,
    ConvertMatMulToGemm,
    ConvertMul,
    ConvertReshapeToFlatten,
    ConvertSubToAdd,
    DropDropout,
    DropIdentity,
    DropUnnecessaryConcat,
    DropUnnecessaryFlatten,
    DropUnnecessaryRelu,
    MoveActivationsBackward,
    SqueezeConvs,
    SqueezeGemms,
]

OPTIONAL_SIMPLIFIERS: typing.Dict[str, typing.Type[Simplifier]] = {
    simplifier_type.__name__: simplifier_type for simplifier_type in (ReluifyMaxPool,)
}


def simplify(
    dnn: OperationGraph, simplifier: typing.Optional[Simplifier] = None
) -> OperationGraph:
    logger = logging.getLogger(__name__)
    if simplifier is None:
        simplifiers = list(DEFAULT_SIMPLIFIERS)
        optional_simplifiers = os.getenv("DNNV_OPTIONAL_SIMPLIFIERS")
        if optional_simplifiers:
            for name in optional_simplifiers.split(":"):
                logger.log(logging.WARNING, "Using optional simplifier: %s", name)
                try:
                    simplifiers.append(OPTIONAL_SIMPLIFIERS[name])
                except KeyError:
                    raise DNNVError(f"Unknown simplifier: {name}")
        simplifier = ComposeSimplifiers(dnn, *simplifiers)
    simplified_graph = OperationGraph(dnn.walk(simplifier))
    return simplified_graph


__all__ = ["simplify", "Simplifier", "ComposeSimplifiers"]
