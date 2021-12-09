from __future__ import annotations

from .base import LTL
from ...base import BinaryExpression


class Release(LTL, BinaryExpression):
    OPERATOR_SYMBOL = "R"


R = Release

__all__ = ["R", "Release"]
