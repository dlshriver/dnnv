import pytest

from dnnv.verifiers.common.reductions.iopolytope import *
from dnnv.verifiers.common.reductions.iopolytope import Constraint, Variable


def setup_function():
    Variable._count = 0


class ConstrainMock(Constraint):
    def update_constraint(self, variables, indices, coefficients, b, is_open):
        return super().update_constraint(
            variables, indices, coefficients, b, is_open=is_open
        )

    def validate(self, *x):
        return super().validate(*x)


def test_isconsistent():
    v = Variable((1, 5))
    c = ConstrainMock(v)
    c.update_constraint(None, None, None, None, None)
    c.validate()
    assert c.is_consistent is None


def test_num_variables():
    v = Variable((1, 5))
    c = ConstrainMock(v)
    assert c.num_variables == 1

    c.add_variable(Variable((1, 3, 2, 2)))
    assert c.num_variables == 2
    c.add_variable(Variable((1, 10)))
    assert c.num_variables == 3


def test_size():
    v = Variable((1, 5))
    c = ConstrainMock(v)
    assert c.size() == 5

    c.add_variable(Variable((1, 3, 2, 2)))
    assert c.size() == 17
    c.add_variable(Variable((1, 10)))
    assert c.size() == 27


def test_add_variable():
    c = ConstrainMock()

    v1 = Variable((1, 5))
    assert v1 not in c.variables
    c.add_variable(v1)
    assert v1 in c.variables
    assert c.num_variables == 1

    v2 = Variable((1, 3, 2, 2))
    c.add_variable(v2)
    assert v2 in c.variables
    assert c.num_variables == 2
    c.add_variable(v2)
    assert c.num_variables == 2

    v3 = Variable((1, 10))
    c.add_variable(v3)
    assert v3 in c.variables
    assert c.num_variables == 3
    c.add_variable(v1)
    assert c.num_variables == 3
    c.add_variable(v3)
    assert c.num_variables == 3


def test_unravel_index():
    v1 = Variable((1, 5))
    v2 = Variable((1, 3, 2, 2))
    v3 = Variable((1, 10))
    c = ConstrainMock(v1)
    c.add_variable(v2)
    c.add_variable(v3)

    assert c.unravel_index(2) == (v1, (0, 2))
    assert c.unravel_index(5) == (v2, (0, 0, 0, 0))
    assert c.unravel_index(15) == (v2, (0, 2, 1, 0))
    assert c.unravel_index(25) == (v3, (0, 8))

    with pytest.raises(
        ValueError,
        match="index 105 is out of bounds for constraint with size 27",
    ):
        _ = c.unravel_index(105)
    with pytest.raises(
        ValueError,
        match="index 27 is out of bounds for constraint with size 27",
    ):
        _ = c.unravel_index(27)
    with pytest.raises(
        ValueError,
        match="index -3 is out of bounds for constraint with size 27",
    ):
        _ = c.unravel_index(-3)
