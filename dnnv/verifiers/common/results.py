from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Collection, Optional


@dataclass
class VerificationResult:
    result: PropertyCheckResult
    results: Collection[PropertyCheckResult] = field(default_factory=tuple, repr=False)
    cex: Optional[Any] = None
    counter_examples: Collection[Any] = field(default_factory=tuple, repr=False)
    vtime: float = 0.0
    vtimes: Collection[float] = field(default_factory=tuple, repr=False)

    def __post_init__(self):
        if len(self.results) == 0:
            self.results = (self.result,)
        if len(self.counter_examples) == 0:
            self.counter_examples = (self.cex,)
        if len(self.vtimes) == 0:
            self.vtimes = (self.vtime,)

    def __or__(self, other):
        if not isinstance(other, VerificationResult):
            return NotImplemented
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
    def __init__(self, name: str):
        self.name = name

    def __str__(self):
        return self.name

    def __repr__(self):
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
