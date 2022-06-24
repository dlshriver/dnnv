"""
dnnv.nn
~~~~~~~
"""
from pathlib import Path
from typing import Optional

from .graph import OperationGraph
from .operations import Operation
from .parser.onnx import parse as parse_onnx
from .transformers import OperationTransformer
from .visitors import OperationVisitor


def try_parse_format(netname: Path) -> str:
    ext = netname.suffix
    if ext == ".onnx":
        return "onnx"
    return ext


def parse(path: Path, net_format: Optional[str] = None) -> OperationGraph:
    if net_format is None:
        net_format = try_parse_format(path)
    if net_format != "onnx":
        raise ValueError(f"Unsupported network format: {net_format}")

    return parse_onnx(path)


__all__ = [
    "Operation",
    "OperationGraph",
    "OperationTransformer",
    "OperationVisitor",
    "parse",
]
