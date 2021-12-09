from __future__ import annotations

from typing import Optional, Union

from .base import LTL, LTLUnwindingBoundReached
from ...base import Expression
from ...terms import Constant, Parameter, Symbol


class TemporalSymbol(LTL):
    def __init__(
        self,
        symbol: Symbol,
        transition_fn: Optional[Expression],
        step: Optional[Union[int, Constant]] = None,
        unwind: Optional[Union[int, Parameter]] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.symbol = symbol
        self.transition_fn = transition_fn
        self.step = step
        if step is None:
            self.step = 0
        elif isinstance(step, Constant):
            self.step = step.value
        self.unwind = unwind
        if unwind is not None and isinstance(unwind, Parameter) and unwind.is_concrete:
            self.unwind = unwind.value

    def __repr__(self):
        transition_fn = ""
        if self.transition_fn is not None:
            transition_fn = f", {self.transition_fn!r}"
        step = ""
        if self.step is not None:
            step = f", {self.step!r}"
        if self.unwind is not None:
            unwind = f", {self.unwind!r}"
        return f"TemporalSymbol({self.symbol!r}{transition_fn}{step}{unwind})"

    def __str__(self):
        step = self.step if self.step is not None else "t"
        return f"{self.symbol}_{step}"

    def freeze(self):
        expr = self.symbol
        for _ in range(self.step):
            expr = self.transition_fn(expr)
        return expr

    def next(self):
        if self.unwind is None:
            raise ValueError("No bound specified for unwinding")
        if self.step < self.unwind:
            return TemporalSymbol(
                self.symbol,
                self.transition_fn,
                step=self.step + 1,
                unwind=self.unwind,
                ctx=self.ctx,
            )
        raise LTLUnwindingBoundReached()


__all__ = ["TemporalSymbol"]
