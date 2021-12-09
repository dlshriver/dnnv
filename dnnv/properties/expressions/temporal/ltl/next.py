from __future__ import annotations

from .base import LTL
from ...base import UnaryExpression


class Next(LTL, UnaryExpression):
    OPERATOR_SYMBOL = "X"


X = Next
__all__ = ["Next", "X"]
