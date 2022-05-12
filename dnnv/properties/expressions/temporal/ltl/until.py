from __future__ import annotations

from ...base import BinaryExpression
from .base import LTL


class Until(LTL, BinaryExpression):
    OPERATOR_SYMBOL = "U"


U = Until

__all__ = ["U", "Until"]
