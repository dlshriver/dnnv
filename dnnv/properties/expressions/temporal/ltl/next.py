from __future__ import annotations

from ...base import UnaryExpression
from .base import LTL


class Next(LTL, UnaryExpression):
    OPERATOR_SYMBOL = "X"


X = Next
__all__ = ["Next", "X"]
