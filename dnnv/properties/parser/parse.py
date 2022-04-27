from pathlib import Path
from typing import List, Optional

from .dnnp import parse as parse_dnnp
from .vnnlib import parse as parse_vnnlib
from ..expressions import Expression


def parse(
    path: Path, format: Optional[str] = None, args: Optional[List[str]] = None
) -> Expression:
    if format == "vnnlib":
        return parse_vnnlib(path, args)
    return parse_dnnp(path, args)


__all__ = ["parse"]
