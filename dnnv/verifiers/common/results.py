from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Collection, Optional


@dataclass
class VerificationResult:
    result: Optional[PropertyCheckResult] = None
    results: Collection[PropertyCheckResult] = field(default_factory=tuple)
    cex: Optional[Any] = None
    counter_examples: Collection[Any] = field(default_factory=tuple)
    vtime: float = 0.0
    vtimes: Collection[float] = field(default_factory=tuple)

    def __post_init__(self):
        if self.result is not None:
            if len(self.results) == 0:
                self.results = (self.result,)
            if len(self.counter_examples) == 0:
                self.counter_examples = (self.cex,)
            if len(self.vtimes) == 0:
                self.vtimes = (self.vtime,)

    def __or__(self, other):
        if not isinstance(other, VerificationResult):
            return NotImplemented
        if self.result is None:
            return other
        result = self.result | other.result
        results = tuple(self.results) + tuple(other.results)
        cex = self.cex if self.cex is not None else other.cex
        counter_examples = tuple(self.counter_examples) + tuple(other.counter_examples)
        vtime = self.vtime + other.vtime
        vtimes = tuple(self.vtimes) + tuple(other.vtimes)
        return VerificationResult(
            result,
            results=results,
            cex=cex,
            counter_examples=counter_examples,
            vtime=vtime,
            vtimes=vtimes,
        )


class PropertyCheckResult:
    def __init__(self, name: str, reason: Optional[str] = None):
        self.name = name
        self.reason = reason

    def __str__(self):
        if self.reason is not None:
            return f"{self.name}({self.reason})"
        return self.name

    def __repr__(self):
        if self.reason is not None:
            return f"PropertyCheckResult({self.name!r}, reason={self.reason!r})"
        return f"PropertyCheckResult({self.name!r})"

    def __invert__(self):
        if self == UNSAT:
            return SAT
        if self == SAT:
            return UNSAT
        return UNKNOWN

    def __and__(self, other):
        if not isinstance(other, PropertyCheckResult):
            return NotImplemented
        if self not in (SAT, UNSAT, UNKNOWN) or other not in (SAT, UNSAT, UNKNOWN):
            return UNKNOWN
        if self == UNSAT or other == UNSAT:
            return UNSAT
        if self == UNKNOWN or other == UNKNOWN:
            return UNKNOWN
        return SAT

    def __or__(self, other):
        if not isinstance(other, PropertyCheckResult):
            return NotImplemented
        if self == SAT or other == SAT:
            return SAT
        if self not in (SAT, UNSAT, UNKNOWN) or other not in (SAT, UNSAT, UNKNOWN):
            return UNKNOWN
        if self == UNKNOWN or other == UNKNOWN:
            return UNKNOWN
        return UNSAT

    def __eq__(self, other):
        if not isinstance(other, PropertyCheckResult):
            return False
        return self.name == other.name


SAT = PropertyCheckResult("sat")
UNKNOWN = PropertyCheckResult("unknown")
UNSAT = PropertyCheckResult("unsat")

__all__ = ["SAT", "UNKNOWN", "UNSAT", "VerificationResult"]
