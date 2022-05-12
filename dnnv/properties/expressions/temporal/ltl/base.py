from __future__ import annotations

from ...logic import LogicalExpression
from ..base import TemporalExpression


class LTL(TemporalExpression, LogicalExpression):
    pass


class LTLMaxStepReached(Exception):
    pass


__all__ = ["LTL", "LTLMaxStepReached"]
