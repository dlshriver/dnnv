import numpy as np
import pytest

from dnnv.verifiers.common.reductions.iopolytope import *
from dnnv.verifiers.common.reductions.iopolytope import Variable


def setup_function():
    Variable._count = 0


def test_update_constraint_single_index():
    v = Variable((1, 3, 2, 2))
    hspoly = HyperRectangle(v)

    variables = [v]
    indices = np.array([(0, 0, 0, 0)])
    coeffs = np.array([1.0])
    b = np.array(10)
    is_open = False
    hspoly.update_constraint(variables, indices, coeffs, b, is_open)
    assert len(hspoly.halfspaces) == 1
    assert np.allclose(hspoly._upper_bound[0], 10)


def test_update_constraint_multiple_indices():
    v = Variable((1, 3, 2, 2))
    hspoly = HyperRectangle(v)

    variables = [v, v]
    indices = np.array([(0, 0, 0, 0), (0, 1, 0, 1)])
    coeffs = np.array([1.0, -1.0])
    b = np.array(10)
    is_open = False
    with pytest.raises(
        ValueError,
        match="HyperRectangle constraints can only constrain a single dimension",
    ):
        hspoly.update_constraint(variables, indices, coeffs, b, is_open)


def test_str():
    v = Variable((1, 5))
    hspoly = HyperRectangle(v)

    variables = [v]
    indices = np.array([(0, 3)])
    coeffs = np.array([-1.0])
    b = np.array(10)
    is_open = False
    hspoly.update_constraint(variables, indices, coeffs, b, is_open)

    variables = [v]
    indices = np.array([(0, 0)])
    coeffs = np.array([1.0])
    b = np.array(5.0)
    is_open = True
    hspoly.update_constraint(variables, indices, coeffs, b, is_open)

    assert str(hspoly) == (
        "-inf <= x_0[(0, 0)] <= 5.000000\n"
        "-inf <= x_0[(0, 1)] <= inf\n"
        "-inf <= x_0[(0, 2)] <= inf\n"
        "-10.000000 <= x_0[(0, 3)] <= inf\n"
        "-inf <= x_0[(0, 4)] <= inf"
    )


def test_is_consistent_true():
    v = Variable((1, 3, 2, 2))
    hspoly = HyperRectangle(v)
    assert hspoly.is_consistent

    variables = [v, v]
    indices = np.array([(0, 0, 0, 0)])
    coeffs = np.array([1.0])
    b = np.array(10)
    is_open = False
    hspoly.update_constraint(variables, indices, coeffs, b, is_open)
    assert hspoly.is_consistent


def test_is_consistent_false():
    v = Variable((1, 3, 2, 2))
    hspoly = HyperRectangle(v)
    assert hspoly.is_consistent

    variables = [v]
    indices = np.array([(0, 0, 0, 0)])
    coeffs = np.array([1.0])
    b = np.array(2)
    is_open = False
    hspoly.update_constraint(variables, indices, coeffs, b, is_open)
    assert hspoly.is_consistent

    variables = [v]
    indices = np.array([(0, 0, 0, 0)])
    coeffs = np.array([-1.0])
    b = np.array(-5)
    is_open = False
    hspoly.update_constraint(variables, indices, coeffs, b, is_open)
    assert not hspoly.is_consistent


def test_lower_bounds():
    v = Variable((1, 5))
    hspoly = HyperRectangle(v)

    variables = [v]
    indices = np.array([(0, 3)])
    coeffs = np.array([-1.0])
    b = np.array(10)
    is_open = False
    hspoly.update_constraint(variables, indices, coeffs, b, is_open)

    variables = [v]
    indices = np.array([(0, 0)])
    coeffs = np.array([1.0])
    b = np.array(5.0)
    is_open = True
    hspoly.update_constraint(variables, indices, coeffs, b, is_open)

    assert len(hspoly.lower_bounds) == 1
    assert np.allclose(
        hspoly.lower_bounds[0], np.array([-np.inf, -np.inf, -np.inf, -10.0, -np.inf])
    )
    hspoly.add_variable(Variable((1, 5)))
    assert len(hspoly.lower_bounds) == 2
    assert np.allclose(
        hspoly.lower_bounds[0], np.array([-np.inf, -np.inf, -np.inf, -10.0, -np.inf])
    )
    assert np.allclose(
        hspoly.lower_bounds[1], np.array([-np.inf, -np.inf, -np.inf, -np.inf, -np.inf])
    )


def test_upper_bounds():
    v = Variable((1, 5))
    hspoly = HyperRectangle(v)

    variables = [v]
    indices = np.array([(0, 3)])
    coeffs = np.array([-1.0])
    b = np.array(10)
    is_open = False
    hspoly.update_constraint(variables, indices, coeffs, b, is_open)

    variables = [v]
    indices = np.array([(0, 0)])
    coeffs = np.array([1.0])
    b = np.array(5.0)
    is_open = True
    hspoly.update_constraint(variables, indices, coeffs, b, is_open)

    assert len(hspoly.upper_bounds) == 1
    assert np.allclose(
        hspoly.upper_bounds[0], np.array([5.0, np.inf, np.inf, np.inf, np.inf])
    )
    hspoly.add_variable(Variable((1, 5)))
    assert len(hspoly.upper_bounds) == 2
    assert np.allclose(
        hspoly.upper_bounds[0], np.array([5.0, np.inf, np.inf, np.inf, np.inf])
    )
    assert np.allclose(
        hspoly.upper_bounds[1], np.array([np.inf, np.inf, np.inf, np.inf, np.inf])
    )
