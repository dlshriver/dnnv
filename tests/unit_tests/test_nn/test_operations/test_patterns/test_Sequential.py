import numpy as np
import pytest

from dnnv.nn.operations import *
from dnnv.nn.operations.patterns import *


def test_init():
    pattern = Sequential(Input, Operation)
    assert isinstance(pattern, Sequential)
    assert len(pattern.patterns) == 2
    assert pattern.patterns[0] == Input
    assert pattern.patterns[1] == Operation

    pattern = Sequential(Operation, Operation)
    assert isinstance(pattern, Sequential)
    assert len(pattern.patterns) == 2
    assert pattern.patterns[0] == Operation
    assert pattern.patterns[1] == Operation


def test_str():
    pattern = Sequential(Input, Operation)
    assert str(pattern) == "(Input >> Operation)"

    pattern = Sequential(Operation, Operation)
    assert str(pattern) == "(Operation >> Operation)"

    pattern = Sequential(Add, Sub, Mul)
    assert str(pattern) == "(Add >> Sub >> Mul)"

    pattern = Sequential(Operation, None)
    assert str(pattern) == "(Operation >> None)"


def test_rshift_error():
    pattern = Sequential(Input, Operation)
    with pytest.raises(TypeError) as excinfo:
        _ = pattern >> 2
    assert str(excinfo.value).startswith(
        "unsupported operand type(s) for >>: 'Sequential' and "
    )

    with pytest.raises(TypeError) as excinfo:
        _ = pattern >> None
    assert str(excinfo.value).startswith(
        "unsupported operand type(s) for >>: 'Sequential' and "
    )


def test_rshift():
    sequential_pattern = Sequential(Input, Operation)

    parallel_pattern = Parallel(Add, Mul)
    pattern = sequential_pattern >> parallel_pattern
    assert isinstance(pattern, Sequential)
    assert len(pattern.patterns) == 3
    assert pattern.patterns[0] == Input
    assert pattern.patterns[1] == Operation
    assert pattern.patterns[2] == parallel_pattern

    or_pattern = Or(Add, Mul)
    pattern = sequential_pattern >> or_pattern
    assert isinstance(pattern, Sequential)
    assert len(pattern.patterns) == 3
    assert pattern.patterns[0] == Input
    assert pattern.patterns[1] == Operation
    assert pattern.patterns[2] == or_pattern

    pattern = sequential_pattern >> Sequential(Mul, Add)
    assert isinstance(pattern, Sequential)
    assert len(pattern.patterns) == 4
    assert pattern.patterns[0] == Input
    assert pattern.patterns[1] == Operation
    assert pattern.patterns[2] == Mul
    assert pattern.patterns[3] == Add


def test_rrshift_error():
    pattern = Sequential(Operation, Input)
    with pytest.raises(TypeError) as excinfo:
        _ = 2 >> pattern
    assert str(excinfo.value).startswith(
        "unsupported operand type(s) for >>: 'int' and 'Sequential'"
    )


def test_rrshift():
    sequential_pattern = Sequential(Input, Operation)

    optional = Or(None)
    # I'm not sure this can actually happen without this explicit call
    # maybe it can be removed, since it's basically the same as the method
    # it overloads from OperationPattern
    pattern = sequential_pattern.__rrshift__(optional)
    assert isinstance(pattern, Sequential)
    assert len(pattern.patterns) == 3
    assert pattern.patterns[0] == optional
    assert pattern.patterns[1] == Input
    assert pattern.patterns[2] == Operation


def test_match_false():
    seq_pattern = Sequential(Input)
    matches = list(seq_pattern.match([Operation()]))
    assert len(matches) == 0

    seq_pattern = Sequential(Add, Sub)
    matches = list(seq_pattern.match([]))
    assert len(matches) == 0
    matches = list(seq_pattern.match([Input(None, None)]))
    assert len(matches) == 0
    matches = list(seq_pattern.match([Mul(None, None)]))
    assert len(matches) == 0
    input_op = Input((), np.dtype(np.float32))
    matches = list(
        seq_pattern.match(
            [
                Mul(input_op, 2.0),
                Div(input_op, 2.0),
            ]
        )
    )
    assert len(matches) == 0
    matches = list(
        seq_pattern.match(
            [
                Add(Mul(input_op, 2.0), 2.0),
            ]
        )
    )
    assert len(matches) == 0


def test_match_true():
    input_op = Input((), np.dtype(np.float32))
    sub_op = Sub(input_op, 2.0)
    add_op = Add(sub_op, 2.0)

    seq_pattern = Sequential(Input)
    matches = list(seq_pattern.match([input_op]))
    assert len(matches) == 1
    assert len(matches[0]) == 0

    seq_pattern = Sequential()
    matches = list(seq_pattern.match([add_op]))
    assert len(matches) == 1
    assert len(matches[0]) == 1
    assert matches[0][0] == add_op

    seq_pattern = Sequential(Add)
    matches = list(seq_pattern.match([add_op]))
    assert len(matches) == 1
    assert len(matches[0]) == 1
    assert matches[0][0] == sub_op

    seq_pattern = Sequential(Sub, Add)
    matches = list(seq_pattern.match([add_op]))
    assert len(matches) == 1
    assert len(matches[0]) == 1
    assert matches[0][0] == input_op

    seq_pattern = Sequential(Input, Sub, Add)
    matches = list(seq_pattern.match([add_op]))
    assert len(matches) == 1
    assert len(matches[0]) == 0
