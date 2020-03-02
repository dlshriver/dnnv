from abc import ABC, abstractmethod
from typing import Optional, Sequence, Union, TYPE_CHECKING

if TYPE_CHECKING:
    from .base import Op, Operation
OpPatternType = Union["Op", "OperationPattern"]


class OperationPattern(ABC):
    @abstractmethod
    def match(self, operations: Sequence["Operation"]):
        raise NotImplementedError()

    def __and__(self, other: Optional[OpPatternType]) -> "Parallel":
        return Parallel(self, other)

    def __rand__(self, other: Optional[OpPatternType]) -> "Parallel":
        return Parallel(other, self)

    def __or__(self, other: Optional[OpPatternType]) -> "Or":
        return Or(self, other)

    def __ror__(self, other: Optional[OpPatternType]) -> "Or":
        return Or(other, self)

    def __rshift__(self, other: OpPatternType) -> "Sequential":
        return Sequential(self, other)

    def __rrshift__(self, other: OpPatternType) -> "Sequential":
        return Sequential(other, self)


class Or(OperationPattern):
    def __init__(self, *patterns: Optional[OpPatternType]):
        self.patterns = set(patterns)

    def __str__(self):
        result_str = " | ".join(str(p) for p in self.patterns)
        return f"({result_str})"

    def __or__(self, other):
        if other is not None and not isinstance(other, OperationPattern):
            return NotImplemented
        if isinstance(other, Or):
            return Or(*self.patterns.union(other.patterns))
        return Or(*self.patterns.union([other]))

    def __ror__(self, other):
        if other is not None and not isinstance(other, OperationPattern):
            return NotImplemented
        return Or(*self.patterns.union([other]))

    def match(self, operations):
        optional = False
        for pattern in self.patterns:
            if pattern is None:
                optional = True
                continue
            for match in pattern.match(operations):
                yield match
        if optional:
            yield operations


class Parallel(OperationPattern):
    def __init__(self, *patterns: Optional[OpPatternType]):
        self.patterns = patterns

    def __str__(self):
        result_str = " & ".join(str(p) for p in self.patterns)
        return f"({result_str})"

    def __and__(self, other):
        if other is not None and not isinstance(other, OperationPattern):
            return NotImplemented
        if isinstance(other, Parallel):
            return Parallel(*(self.patterns + other.patterns))
        return Parallel(*(self.patterns + (other,)))

    def __rand__(self, other):
        if other is not None and not isinstance(other, OperationPattern):
            return NotImplemented
        return Parallel(*((other,) + self.patterns))

    def match(self, operations):
        if len(operations) != len(self.patterns):
            return
        matches = [[]]
        for pattern, operation in zip(self.patterns, operations):
            if pattern is None:
                for match in matches:
                    match.append(operation)
                continue
            new_matches = []
            for new_match in pattern.match([operation]):
                for match in matches:
                    new_matches.append(match + new_match)
            matches = new_matches
        for match in matches:
            match_set = set(match)
            if len(match_set) == 1:
                yield list(match_set)
            elif len(match_set) == len(match):
                yield match
            else:
                raise AssertionError(
                    "Unexpected error: Parallel match was not length 1 or N"
                )  # impossible?


class Sequential(OperationPattern):
    def __init__(self, *patterns: OpPatternType):
        self.patterns = patterns

    def __str__(self):
        result_str = " >> ".join(str(p) for p in self.patterns)
        return f"({result_str})"

    def __rshift__(self, other):
        if not isinstance(other, OperationPattern):
            return NotImplemented
        if isinstance(other, Sequential):
            return Sequential(*(self.patterns + other.patterns))
        return Sequential(*(self.patterns + (other,)))

    def __rrshift__(self, other):
        if not isinstance(other, OperationPattern):
            return NotImplemented
        return Sequential(*((other,) + self.patterns))

    def match(self, operations):
        next_operations = [operations]
        for pattern in reversed(self.patterns):
            matches = []
            for ops in next_operations:
                for match in pattern.match(ops):
                    matches.append(match)
            next_operations = matches
        for match in next_operations:
            yield match
