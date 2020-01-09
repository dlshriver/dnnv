class PropertyCheckResult:
    def __init__(self, name):
        self.name = name

    def __str__(self):
        return self.name

    def __invert__(self):
        if self == UNSAT:
            return SAT
        if self == SAT:
            return UNSAT
        return UNKNOWN

    def __and__(self, other):
        if not isinstance(other, PropertyCheckResult):
            return NotImplemented
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
        if self == UNKNOWN or other == UNKNOWN:
            return UNKNOWN
        return UNSAT


SAT = PropertyCheckResult("sat")
UNKNOWN = PropertyCheckResult("unknown")
UNSAT = PropertyCheckResult("unsat")

__all__ = ["SAT", "UNKNOWN", "UNSAT"]
