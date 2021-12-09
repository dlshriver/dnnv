from __future__ import annotations

from .base import LTL
from ...base import UnaryExpression


class Finally(LTL, UnaryExpression):
    OPERATOR_SYMBOL = "F"


F = Finally
__all__ = ["Finally", "F"]
