from __future__ import annotations

from ..base import TemporalExpression
from ...logic import LogicalExpression


class LTL(TemporalExpression, LogicalExpression):
    pass


class LTLMaxStepReached(Exception):
    pass


__all__ = ["LTL", "LTLMaxStepReached"]
