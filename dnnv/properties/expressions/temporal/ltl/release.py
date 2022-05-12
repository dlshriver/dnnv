from __future__ import annotations

from ...base import BinaryExpression
from .base import LTL


class Release(LTL, BinaryExpression):
    OPERATOR_SYMBOL = "R"


R = Release

__all__ = ["R", "Release"]
