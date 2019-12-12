from abc import ABC, abstractmethod


class OpPatternMatch:
    def __init__(self, matching_operation_graph, input_operations):
        pass


class OperationPattern(ABC):
    def __init__(self, pattern=None):
        self.pattern = pattern

    @abstractmethod
    def match(self, operations):
        pass

    def __and__(self, other):
        return Parallel(self, other)

    def __rand__(self, other):
        return Parallel(other, self)

    def __or__(self, other):
        return Or(self, other)

    def __ror__(self, other):
        return Or(other, self)

    def __rshift__(self, other):
        return Sequential(self, other)

    def __rrshift__(self, other):
        return Sequential(other, self)


class Or(OperationPattern):
    def __init__(self, *patterns):
        self.patterns = set(patterns)

    def __str__(self):
        result_str = " | ".join(str(p) for p in self.patterns)
        return f"({result_str})"

    def __or__(self, other):
        if isinstance(other, Or):
            return Or(*self.patterns.union(other.patterns))
        return Or(*self.patterns.union([other]))

    def __ror__(self, other):
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
    def __init__(self, *patterns):
        self.patterns = patterns

    def __str__(self):
        result_str = " & ".join(str(p) for p in self.patterns)
        return f"({result_str})"

    def __and__(self, other):
        if isinstance(other, Parallel):
            return Parallel(*(self.patterns + other.patterns))
        return Parallel(*(self.patterns + (other,)))

    def __rand__(self, other):
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
    def __init__(self, *patterns):
        self.patterns = patterns

    def __str__(self):
        result_str = " >> ".join(str(p) for p in self.patterns)
        return f"({result_str})"

    def __rshift__(self, other):
        if isinstance(other, Sequential):
            return Sequential(*(self.patterns + other.patterns))
        return Sequential(*(self.patterns + (other,)))

    def __rrshift__(self, other):
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

