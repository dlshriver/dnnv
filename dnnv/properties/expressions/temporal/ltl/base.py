from __future__ import annotations

from ..base import TemporalExpression
from ...logic import LogicalExpression


class LTL(TemporalExpression, LogicalExpression):
    pass


class LTLUnwindingBoundReached(Exception):
    pass


__all__ = ["LTL", "LTLUnwindingBoundReached"]
