from __future__ import annotations

from .base import LTL
from ...base import BinaryExpression


class StrongRelease(LTL, BinaryExpression):
    OPERATOR_SYMBOL = "M"


M = StrongRelease

__all__ = ["M", "StrongRelease"]
