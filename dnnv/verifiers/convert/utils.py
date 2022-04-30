from __future__ import annotations

import tempfile
from typing import Iterable, Optional, Type, Union

from dnnv.nn.graph import OperationGraph
from dnnv.verifiers.common import (
    HalfspacePolytope,
    IOPolytopeProperty,
    VerifierTranslatorError,
)


def as_vnnlib(
    prop: IOPolytopeProperty,
    translator_error: Type[VerifierTranslatorError] = VerifierTranslatorError,
    *,
    extended_vnnlib=False,
) -> Iterable[str]:
    for i in range(prop.input_constraint.size()):
        if extended_vnnlib:
            _, index = prop.input_constraint.unravel_index(i)
            index_str = "_".join(str(d) for d in index)
        else:
            index_str = str(i)
        yield f"(declare-const X_{index_str} Real)"
    for i in range(prop.output_constraint.size()):
        if extended_vnnlib:
            _, index = prop.output_constraint.unravel_index(i)
            index_str = "_".join(str(d) for d in index)
        else:
            index_str = str(i)
        yield f"(declare-const Y_{index_str} Real)"
    assert isinstance(
        prop.input_constraint, HalfspacePolytope
    ), "Input constraints expected to be represented as a halfspace-polytope"
    for halfspace in prop.input_constraint.halfspaces:
        summands = []
        for c, i in zip(halfspace.coefficients, halfspace.indices):
            if extended_vnnlib:
                _, index = prop.input_constraint.unravel_index(i)
                index_str = "_".join(str(d) for d in index)
            else:
                index_str = str(i)
            if c == 0:
                continue
            elif c == 1:
                summands.append(f"X_{index_str}")
            elif c == -1:
                summands.append(f"(- X_{index_str})")
            elif c >= 0:
                summands.append(f"(* {c:.12f} X_{index_str})")
            else:
                summands.append(f"(* (- {abs(c):.12f}) X_{index_str})")
        if len(summands) == 0:
            lhs: Union[float, str] = 0.0
        elif len(summands) == 1:
            lhs = summands[0]
        else:
            summands_str = " ".join(summands)
            lhs = f"(+ {summands_str})"
        yield f"(assert (<= {lhs} {halfspace.b}))"
    assert isinstance(
        prop.output_constraint, HalfspacePolytope
    ), "Output constraints expected to be represented as a halfspace-polytope"
    for halfspace in prop.output_constraint.halfspaces:
        summands = []
        for c, i in zip(halfspace.coefficients, halfspace.indices):
            if extended_vnnlib:
                _, index = prop.output_constraint.unravel_index(i)
                index_str = "_".join(str(d) for d in index)
            else:
                index_str = str(i)
            if c == 0:
                continue
            elif c == 1:
                summands.append(f"Y_{index_str}")
            elif c == -1:
                summands.append(f"(- Y_{index_str})")
            elif c >= 0:
                summands.append(f"(* {c:.12f} Y_{index_str})")
            else:
                summands.append(f"(* (- {abs(c):.12f}) Y_{index_str})")
        if len(summands) == 0:
            lhs = 0.0
        elif len(summands) == 1:
            lhs = summands[0]
        else:
            summands_str = " ".join(summands)
            lhs = f"(+ {summands_str})"
        yield f"(assert (<= {lhs} {halfspace.b}))"


def to_vnnlib_property_file(
    prop: IOPolytopeProperty,
    dirname: Optional[str] = None,
    translator_error: Type[VerifierTranslatorError] = VerifierTranslatorError,
    *,
    extended_vnnlib=False,
) -> str:
    with tempfile.NamedTemporaryFile(
        mode="w+", dir=dirname, suffix=".vnnlib", delete=False
    ) as vnnlib_file:
        for line in as_vnnlib(
            prop,
            translator_error=translator_error,
            extended_vnnlib=extended_vnnlib,
        ):
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


__all__ = ["as_vnnlib", "to_vnnlib_onnx_file", "to_vnnlib_property_file"]
