from __future__ import annotations

from .base import LTL
from ...base import UnaryExpression


class Globally(LTL, UnaryExpression):
    OPERATOR_SYMBOL = "G"


G = Globally
__all__ = ["Globally", "G"]
