"""
dnnv.nn
~~~~~~~
"""
from pathlib import Path
from typing import Optional

from .graph import OperationGraph
from .operations import Operation
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
    if net_format == "onnx":
        from .parser.onnx import parse as _parse
    else:
        raise ValueError("Unsupported network format: %s" % net_format)

    return _parse(path)
