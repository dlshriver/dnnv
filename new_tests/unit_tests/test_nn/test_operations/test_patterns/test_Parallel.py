import numpy as np
import pytest

from dnnv.nn.operations import *
from dnnv.nn.operations.patterns import *


def test_init():
    pattern = Parallel(Operation, Input)
    assert isinstance(pattern, Parallel)
    assert len(pattern.patterns) == 2
    assert pattern.patterns[0] == Operation
    assert pattern.patterns[1] == Input

    pattern = Parallel(Operation, Operation)
    assert isinstance(pattern, Parallel)
    assert len(pattern.patterns) == 2
    assert pattern.patterns[0] == Operation
    assert pattern.patterns[1] == Operation


def test_str():
    pattern = Parallel(Operation, Input)
    assert str(pattern) == "(Operation & Input)"

    pattern = Parallel(Operation, Operation)
    assert str(pattern) == "(Operation & Operation)"

    pattern = Parallel(Add, Sub, Mul)
    assert str(pattern) == "(Add & Sub & Mul)"

    pattern = Parallel(Operation, None)
    assert str(pattern) == "(Operation & None)"


def test_and_error():
    pattern = Parallel(Operation, Input)
    with pytest.raises(TypeError) as excinfo:
        _ = pattern & 2
    assert str(excinfo.value).startswith(
        "unsupported operand type(s) for &: 'Parallel' and "
    )


def test_and():
    parallel_pattern = Parallel(Operation, Input)

    pattern = parallel_pattern & None
    assert isinstance(pattern, Parallel)
    assert len(pattern.patterns) == 3
    assert pattern.patterns[0] == Operation
    assert pattern.patterns[1] == Input
    assert pattern.patterns[2] == None

    pattern = parallel_pattern & Parallel(Add, Mul)
    assert isinstance(pattern, Parallel)
    assert len(pattern.patterns) == 4
    assert pattern.patterns[0] == Operation
    assert pattern.patterns[1] == Input
    assert pattern.patterns[2] == Add
    assert pattern.patterns[3] == Mul

    or_pattern = Or(Add, Mul)
    pattern = parallel_pattern & or_pattern
    assert isinstance(pattern, Parallel)
    assert len(pattern.patterns) == 3
    assert pattern.patterns[0] == Operation
    assert pattern.patterns[1] == Input
    assert pattern.patterns[2] == or_pattern

    sequential_pattern = Sequential(Add, Mul)
    pattern = parallel_pattern & sequential_pattern
    assert isinstance(pattern, Parallel)
    assert len(pattern.patterns) == 3
    assert pattern.patterns[0] == Operation
    assert pattern.patterns[1] == Input
    assert pattern.patterns[2] == sequential_pattern


def test_rand_error():
    pattern = Parallel(Operation, Input)
    with pytest.raises(TypeError) as excinfo:
        _ = 2 & pattern
    assert str(excinfo.value).startswith(
        "unsupported operand type(s) for &: 'int' and 'Parallel'"
    )


def test_ror():
    parallel_pattern = Parallel(Operation, Input)

    pattern = None & parallel_pattern
    assert isinstance(pattern, Parallel)
    assert len(pattern.patterns) == 3
    assert pattern.patterns[0] == None
    assert pattern.patterns[1] == Operation
    assert pattern.patterns[2] == Input


def test_match_false():
    parallel_pattern_empty = Parallel()
    matches = list(parallel_pattern_empty.match([Operation()]))
    assert len(matches) == 0

    par_pattern = Parallel(Add, Sub)
    matches = list(par_pattern.match([]))
    assert len(matches) == 0
    matches = list(par_pattern.match([Input(None, None)]))
    assert len(matches) == 0
    matches = list(par_pattern.match([Mul(None, None)]))
    assert len(matches) == 0
    input_op = Input((), np.dtype(np.float32))
    matches = list(
        par_pattern.match(
            [
                Mul(input_op, 2.0),
                Div(input_op, 2.0),
            ]
        )
    )
    assert len(matches) == 0


def test_match_true():
    input_op = Input((), np.dtype(np.float32))

    par_pattern = Parallel(Input)
    matches = list(par_pattern.match([input_op]))
    assert len(matches) == 1
    assert len(matches[0]) == 0

    par_pattern = Parallel(Add, Sub)
    matches = list(
        par_pattern.match(
            [
                Add(input_op, 2.0),
                Sub(input_op, 2.0),
            ]
        )
    )
    assert len(matches) == 1
    assert len(matches[0]) == 1
    assert matches[0][0] == input_op

    par_pattern = Parallel(Add, Sub)
    _input_op = Input((), np.dtype(np.float32))
    matches = list(
        par_pattern.match(
            [
                Add(input_op, 2.0),
                Sub(_input_op, 2.0),
            ]
        )
    )
    assert len(matches) == 1
    assert len(matches[0]) == 2
    assert matches[0][0] == input_op
    assert matches[0][1] == _input_op


def test_match_optional():
    input_op = Input((), np.dtype(np.float32))
    add_op = Add(input_op, 2.0)
    mul_op = Mul(input_op, 2.0)
    sub_op = Sub(input_op, 2.0)

    par_pattern = Parallel(None)
    matches = list(par_pattern.match([add_op]))
    assert len(matches) == 1
    assert matches[0][0] == add_op
    matches = list(par_pattern.match([mul_op]))
    assert len(matches) == 1
    assert matches[0][0] == mul_op

    par_pattern = Parallel(Sub, None)
    matches = list(par_pattern.match([sub_op, add_op]))
    assert len(matches) == 1
    assert len(matches[0]) == 2
    assert matches[0][0] == input_op
    assert matches[0][1] == add_op
    matches = list(par_pattern.match([sub_op, mul_op]))
    assert len(matches) == 1
    assert len(matches[0]) == 2
    assert matches[0][0] == input_op
    assert matches[0][1] == mul_op
    matches = list(par_pattern.match([sub_op, input_op]))
    assert len(matches) == 1
    assert len(matches[0]) == 1
    assert matches[0][0] == input_op
