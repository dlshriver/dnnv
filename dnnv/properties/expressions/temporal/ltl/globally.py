from __future__ import annotations

from ...base import UnaryExpression
from .base import LTL


class Globally(LTL, UnaryExpression):
    OPERATOR_SYMBOL = "G"


G = Globally
__all__ = ["Globally", "G"]
