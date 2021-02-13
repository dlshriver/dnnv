import tempfile

from dnnv.nn.graph import OperationGraph
from dnnv.verifiers.common import (
    HalfspacePolytope,
    IOPolytopeProperty,
    VerifierTranslatorError,
)
from typing import Iterable, Optional, Type


def as_vnnlib(
    prop: IOPolytopeProperty,
    translator_error: Type[VerifierTranslatorError] = VerifierTranslatorError,
) -> Iterable[str]:
    for i in range(prop.input_constraint.size()):
        yield f"(declare-const X_{i} Real)"
    for i in range(prop.output_constraint.size()):
        yield f"(declare-const Y_{i} Real)"
    assert isinstance(
        prop.input_constraint, HalfspacePolytope
    ), "Input constraints expected to be represented as a halfspace-polytope"
    for halfspace in prop.input_constraint.halfspaces:
        summands = " ".join(
            f"(* {c:.12f} X_{i})" if c >= 0 else f"(* (- {abs(c):.12f}) X_{i})"
            for c, i in zip(halfspace.coefficients, halfspace.indices)
        )
        yield f"(assert (<= (+ {summands}) {halfspace.b}))"
    assert isinstance(
        prop.output_constraint, HalfspacePolytope
    ), "Output constraints expected to be represented as a halfspace-polytope"
    for halfspace in prop.output_constraint.halfspaces:
        summands = " ".join(
            f"(* {c:.12f} Y_{i})" if c >= 0 else f"(* (- {abs(c):.12f}) Y_{i})"
            for c, i in zip(halfspace.coefficients, halfspace.indices)
        )
        yield f"(assert (<= (+ {summands}) {halfspace.b}))"


def to_vnnlib_property_file(
    prop: IOPolytopeProperty,
    dirname: Optional[str] = None,
    translator_error: Type[VerifierTranslatorError] = VerifierTranslatorError,
) -> str:
    with tempfile.NamedTemporaryFile(
        mode="w+", dir=dirname, suffix=".vnnlib", delete=False
    ) as vnnlib_file:
        for line in as_vnnlib(prop, translator_error=translator_error):
            vnnlib_file.write(f"{line}\n")
        return vnnlib_file.name


def to_vnnlib_onnx_file(
    dnn: OperationGraph,
    dirname: Optional[str] = None,
    translator_error: Type[VerifierTranslatorError] = VerifierTranslatorError,
) -> str:
    with tempfile.NamedTemporaryFile(
        mode="w+", dir=dirname, suffix=".onnx", delete=False
    ) as onnx_file:
        dnn.export_onnx(onnx_file.name)
        return onnx_file.name
