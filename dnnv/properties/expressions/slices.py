from __future__ import annotations

from typing import Any, Optional

from .base import AssociativeExpression, Expression, TernaryExpression
from .context import Context
from .terms import Constant


class Slice(TernaryExpression):
    def __init__(
        self,
        start: Any,
        stop: Any,
        step: Any,
        *,
        ctx: Optional[Context] = None,
    ):
        start = start if isinstance(start, Expression) else Constant(start)
        stop = stop if isinstance(stop, Expression) else Constant(stop)
        step = step if isinstance(step, Expression) else Constant(step)
        super().__init__(start, stop, step, ctx=ctx)

    def OPERATOR(self, start, stop, step):
        return slice(start, stop, step)

    @property
    def start(self):
        return self.expr1

    @property
    def stop(self):
        return self.expr2

    @property
    def step(self):
        return self.expr3

    def __repr__(self):
        if self.step is None or (self.step.is_concrete and self.step.value is None):
            return f"{self.start!r}:{self.stop!r}"
        return f"{self.start!r}:{self.stop!r}:{self.step!r}"

    def __str__(self):
        if self.step is None or (self.step.is_concrete and self.step.value is None):
            return f"{self.start}:{self.stop}"
        return f"{self.start}:{self.stop}:{self.step}"


class ExtSlice(AssociativeExpression):
    BASE_VALUE = ()
    OPERATOR_SYMBOL = ","

    @property
    def value(self):
        if len(self.expressions) > 0:
            return tuple(expr.value for expr in self.expressions)
        return ()


__all__ = ["ExtSlice", "Slice"]
