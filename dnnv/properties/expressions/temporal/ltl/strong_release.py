from __future__ import annotations

from ...base import BinaryExpression
from .base import LTL


class StrongRelease(LTL, BinaryExpression):
    OPERATOR_SYMBOL = "M"


M = StrongRelease

__all__ = ["M", "StrongRelease"]
