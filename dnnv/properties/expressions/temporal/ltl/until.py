from __future__ import annotations

from .base import LTL
from ...base import BinaryExpression


class Until(LTL, BinaryExpression):
    OPERATOR_SYMBOL = "U"


U = Until

__all__ = ["U", "Until"]
