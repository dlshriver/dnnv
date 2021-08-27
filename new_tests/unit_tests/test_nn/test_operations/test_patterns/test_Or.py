import numpy as np
import pytest

from dnnv.nn.operations import *
from dnnv.nn.operations.patterns import *


def test_init():
    or_pattern = Or(Operation, Input)
    assert isinstance(or_pattern, Or)
    assert len(or_pattern.patterns) == 2
    assert Operation in or_pattern.patterns
    assert Input in or_pattern.patterns

    or_pattern = Or(Operation, Operation)
    assert isinstance(or_pattern, Or)
    assert len(or_pattern.patterns) == 1
    assert Operation in or_pattern.patterns


def test_str():
    or_pattern = Or(Operation, Input)
    assert str(or_pattern) in ("(Operation | Input)", "(Input | Operation)")

    or_pattern = Or(Operation, Operation)
    assert str(or_pattern) == "(Operation)"

    or_pattern = Or(Add, Sub, Mul)
    assert str(or_pattern) in (
        "(Add | Sub | Mul)",
        "(Mul | Add | Sub)",
        "(Sub | Mul | Add)",
        "(Sub | Add | Mul)",
        "(Mul | Sub | Add)",
        "(Add | Mul | Sub)",
    )

    or_pattern = Or(Operation, None)
    assert str(or_pattern) in ("(Operation | None)", "(None | Operation)")


def test_or_error():
    or_pattern = Or(Operation, Input)
    with pytest.raises(TypeError) as excinfo:
        _ = or_pattern | 2
    assert str(excinfo.value).startswith("unsupported operand type(s) for |: 'Or' and ")


def test_or():
    or_pattern = Or(Operation, Input)

    pattern = or_pattern | None
    assert isinstance(pattern, Or)
    assert len(pattern.patterns) == 3
    assert None in pattern.patterns
    assert Operation in pattern.patterns
    assert Input in pattern.patterns

    pattern = or_pattern | Or(Add, Mul)
    assert isinstance(pattern, Or)
    assert len(pattern.patterns) == 4
    assert Add in pattern.patterns
    assert Mul in pattern.patterns
    assert Operation in pattern.patterns
    assert Input in pattern.patterns

    parallel_pattern = Parallel(Add, Mul)
    pattern = or_pattern | parallel_pattern
    assert isinstance(pattern, Or)
    assert len(pattern.patterns) == 3
    assert parallel_pattern in pattern.patterns
    assert Operation in pattern.patterns
    assert Input in pattern.patterns

    sequential_pattern = Sequential(Add, Mul)
    pattern = or_pattern | sequential_pattern
    assert isinstance(pattern, Or)
    assert len(pattern.patterns) == 3
    assert sequential_pattern in pattern.patterns
    assert Operation in pattern.patterns
    assert Input in pattern.patterns


def test_ror_error():
    or_pattern = Or(Operation, Input)
    with pytest.raises(TypeError) as excinfo:
        _ = 2 | or_pattern
    assert str(excinfo.value).startswith(
        "unsupported operand type(s) for |: 'int' and 'Or'"
    )


def test_ror():
    or_pattern = Or(Operation, Input)

    pattern = None | or_pattern
    assert isinstance(pattern, Or)
    assert len(pattern.patterns) == 3
    assert None in pattern.patterns
    assert Operation in pattern.patterns
    assert Input in pattern.patterns


def test_match_false():
    or_pattern_empty = Or()
    matches = list(or_pattern_empty.match([Operation()]))
    assert len(matches) == 0

    or_pattern = Or(Add, Sub)
    matches = list(or_pattern.match([Input(None, None)]))
    assert len(matches) == 0
    matches = list(or_pattern.match([Mul(None, None)]))
    assert len(matches) == 0
    matches = list(
        or_pattern.match(
            [
                Add(Input((), np.dtype(np.float32)), 2.0),
                Sub(Input((), np.dtype(np.float32)), 2.0),
            ]
        )
    )
    assert len(matches) == 0


def test_match_true():
    or_pattern = Or(Add, Sub)

    input_op = Input((), np.dtype(np.float32))
    matches = list(or_pattern.match([Add(input_op, 2.0)]))
    assert len(matches) == 1
    assert matches[0][0] == input_op


def test_match_optional():
    input_op = Input((), np.dtype(np.float32))

    or_pattern = Or(None)
    add_op = Add(input_op, 2.0)
    matches = list(or_pattern.match([add_op]))
    assert len(matches) == 1
    assert matches[0][0] == add_op
    mul_op = Mul(input_op, 2.0)
    matches = list(or_pattern.match([mul_op]))
    assert len(matches) == 1
    assert matches[0][0] == mul_op

    or_pattern = Or(Sub, None)
    matches = list(or_pattern.match([add_op]))
    assert len(matches) == 1
    assert matches[0][0] == add_op
    matches = list(or_pattern.match([mul_op]))
    assert len(matches) == 1
    assert matches[0][0] == mul_op
    matches = list(or_pattern.match([input_op]))
    assert len(matches) == 1
    assert matches[0][0] == input_op
