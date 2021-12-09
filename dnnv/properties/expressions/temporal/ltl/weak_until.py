from __future__ import annotations

from .base import LTL
from ...base import BinaryExpression


class WeakUntil(LTL, BinaryExpression):
    OPERATOR_SYMBOL = "W"


W = WeakUntil

__all__ = ["W", "WeakUntil"]
