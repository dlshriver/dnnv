from __future__ import annotations

from ...base import UnaryExpression
from .base import LTL


class Finally(LTL, UnaryExpression):
    OPERATOR_SYMBOL = "F"


F = Finally
__all__ = ["Finally", "F"]
