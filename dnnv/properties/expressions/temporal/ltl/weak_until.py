from __future__ import annotations

from ...base import BinaryExpression
from .base import LTL


class WeakUntil(LTL, BinaryExpression):
    OPERATOR_SYMBOL = "W"


W = WeakUntil

__all__ = ["W", "WeakUntil"]
