from pathlib import Path
from typing import List, Optional

from .base import Expression
from .dsl import parse as parse_dsl
from .vnnlib import parse as parse_vnnlib


def parse(
    path: Path, format: Optional[str] = None, args: Optional[List[str]] = None
) -> Expression:
    if format == "vnnlib":
        return parse_vnnlib(path, args)
    return parse_dsl(path, args)
