import numpy as np

from dnnv.nn.graph import OperationGraph
from dnnv.nn import operations
from dnnv.properties.expressions import *
from dnnv.verifiers.common.reductions.iopolytope import *
from dnnv.verifiers.common.reductions.iopolytope import Variable


def setup_function():
    Variable._count = 0


def test_add_variable():
    v1 = Variable((1, 3, 2, 2))
    hspoly = HalfspacePolytope()
    assert hspoly.size() == 0
    hspoly.add_variable(v1)
    assert hspoly.size() == 12
    v2 = Variable((1, 5))
    hspoly.add_variable(v2)
    assert hspoly.size() == 17


def test_update_constraint_single_index():
    v = Variable((1, 3, 2, 2))
    hspoly = HalfspacePolytope(v)

    variables = [v]
    indices = np.array([(0, 0, 0, 0)])
    coeffs = np.array([1.0])
    b = np.array(10)
    is_open = False
    hspoly.update_constraint(variables, indices, coeffs, b, is_open)
    assert len(hspoly.halfspaces) == 1
    assert np.allclose(hspoly._upper_bound[0], 10)


def test_update_constraint_single_index_open():
    v = Variable((1, 3, 2, 2))
    hspoly = HalfspacePolytope(v)

    variables = [v]
    indices = np.array([(0, 0, 0, 0)])
    coeffs = np.array([1.0])
    b = np.array(10)
    is_open = True
    hspoly.update_constraint(variables, indices, coeffs, b, is_open)
    assert len(hspoly.halfspaces) == 1
    assert np.allclose(hspoly._upper_bound[0], 10)
    assert np.all(hspoly._upper_bound[0] < 10)


def test_update_constraint_single_index_negative_coeff():
    v = Variable((1, 3, 2, 2))
    hspoly = HalfspacePolytope(v)

    variables = [v]
    indices = np.array([(0, 0, 0, 0)])
    coeffs = np.array([-1.0])
    b = np.array(10)
    is_open = False
    hspoly.update_constraint(variables, indices, coeffs, b, is_open)
    assert len(hspoly.halfspaces) == 1
    assert np.allclose(hspoly._lower_bound[0], -10)


def test_update_constraint_single_index_negative_coeff_is_open():
    v = Variable((1, 3, 2, 2))
    hspoly = HalfspacePolytope(v)

    variables = [v]
    indices = np.array([(0, 0, 0, 0)])
    coeffs = np.array([-1.0])
    b = np.array(10)
    is_open = True
    hspoly.update_constraint(variables, indices, coeffs, b, is_open)
    assert len(hspoly.halfspaces) == 1
    assert np.allclose(hspoly._lower_bound[0], -10)
    assert np.all(hspoly._lower_bound[0] > -10)


def test_update_constraint_multiple_indices():
    v = Variable((1, 3, 2, 2))
    hspoly = HalfspacePolytope(v)

    variables = [v, v]
    indices = np.array([(0, 0, 0, 0), (0, 1, 0, 1)])
    coeffs = np.array([1.0, -1.0])
    b = np.array(10)
    is_open = False
    hspoly.update_constraint(variables, indices, coeffs, b, is_open)
    assert len(hspoly.halfspaces) == 1
    assert len(hspoly._A) == 1
    assert np.allclose(hspoly._A[0], np.array([1.0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0]))
    assert np.allclose(hspoly._b[0], 10)


def test_update_constraint_multiple_indices_is_open():
    v = Variable((1, 3, 2, 2))
    hspoly = HalfspacePolytope(v)

    variables = [v, v]
    indices = np.array([(0, 0, 0, 0), (0, 1, 0, 1)])
    coeffs = np.array([1.0, -1.0])
    b = np.array(10)
    is_open = True
    hspoly.update_constraint(variables, indices, coeffs, b, is_open)
    assert len(hspoly.halfspaces) == 1
    assert len(hspoly._A) == 1
    assert np.allclose(hspoly._A[0], np.array([1.0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0]))
    assert np.allclose(hspoly._b[0], 10)
    assert hspoly._b[0] < 10


def test_update_constraint_closed_polytope():
    v = Variable((1, 2))
    hspoly = HalfspacePolytope(v)

    variables = [v, v]
    indices = np.array([(0, 0), (0, 1)])
    coeffs = np.array([1.0, 1.0])
    b = np.array(2)
    hspoly.update_constraint(variables, indices, coeffs, b)

    variables = [v, v]
    indices = np.array([(0, 0), (0, 1)])
    coeffs = np.array([-1.0, -1.0])
    b = np.array(2)
    hspoly.update_constraint(variables, indices, coeffs, b)

    variables = [v, v]
    indices = np.array([(0, 0), (0, 1)])
    coeffs = np.array([1.0, -1.0])
    b = np.array(4)
    hspoly.update_constraint(variables, indices, coeffs, b)

    variables = [v, v]
    indices = np.array([(0, 0), (0, 1)])
    coeffs = np.array([-1.0, 1.0])
    b = np.array(11)
    hspoly.update_constraint(variables, indices, coeffs, b)

    assert hspoly.is_consistent

    assert np.allclose(hspoly._lower_bound, np.array([-6.5, -3.0]))
    assert np.allclose(hspoly._upper_bound, np.array([3.0, 6.5]))


def test_validate_bad_data():
    v = Variable((1, 3, 2, 2))
    hspoly = HalfspacePolytope(v)

    assert not hspoly.validate()
    assert not hspoly.validate(np.ones((1, 3)))


def test_validate_true():
    v = Variable((1, 3, 2, 2))
    hspoly = HalfspacePolytope(v)

    variables = [v, v]
    indices = np.array([(0, 0, 0, 0), (0, 1, 0, 1)])
    coeffs = np.array([1.0, -1.0])
    b = np.array(10)
    is_open = False
    hspoly.update_constraint(variables, indices, coeffs, b, is_open)

    x = np.zeros((1, 3, 2, 2))
    x[0, 0, 0, 0] = 1
    x[0, 1, 0, 1] = 3
    assert hspoly.validate(x)

    variables = [v]
    indices = np.array([(0, 0, 0, 0)])
    coeffs = np.array([1.0])
    b = np.array(5)
    is_open = True
    hspoly.update_constraint(variables, indices, coeffs, b, is_open)

    x = np.zeros((1, 3, 2, 2))
    x[0, 0, 0, 0] = 1
    x[0, 1, 0, 1] = 3
    assert hspoly.validate(x)


def test_validate_false():
    v = Variable((1, 3, 2, 2))
    hspoly = HalfspacePolytope(v)

    variables = [v, v]
    indices = np.array([(0, 0, 0, 0), (0, 1, 0, 1)])
    coeffs = np.array([1.0, -1.0])
    b = np.array(10)
    is_open = False
    hspoly.update_constraint(variables, indices, coeffs, b, is_open)

    x = np.zeros((1, 3, 2, 2))
    x[0, 0, 0, 0] = 100
    x[0, 1, 0, 1] = 3
    assert not hspoly.validate(x)


def test_str():
    v = Variable((1, 3, 2, 2))
    hspoly = HalfspacePolytope(v)

    variables = [v, v]
    indices = np.array([(0, 0, 0, 0), (0, 1, 0, 1)])
    coeffs = np.array([1.0, -1.0])
    b = np.array(10)
    is_open = False
    hspoly.update_constraint(variables, indices, coeffs, b, is_open)

    variables = [v]
    indices = np.array([(0, 0, 0, 0)])
    coeffs = np.array([1.0])
    b = np.array(5.0)
    is_open = True
    hspoly.update_constraint(variables, indices, coeffs, b, is_open)

    assert (
        str(hspoly)
        == "1.0 * x_0[(0, 0, 0, 0)] + -1.0 * x_0[(0, 1, 0, 1)] <= 10\n1.0 * x_0[(0, 0, 0, 0)] < 5.0"
    )


def test_is_consistent_true():
    v = Variable((1, 3, 2, 2))
    hspoly = HalfspacePolytope(v)
    assert hspoly.is_consistent

    variables = [v, v]
    indices = np.array([(0, 0, 0, 0), (0, 1, 0, 1)])
    coeffs = np.array([1.0, -1.0])
    b = np.array(10)
    is_open = False
    hspoly.update_constraint(variables, indices, coeffs, b, is_open)
    assert hspoly.is_consistent


def test_is_consistent_false():
    v = Variable((1, 3, 2, 2))
    hspoly = HalfspacePolytope(v)
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


def test_as_matrix_inequality():
    v = Variable((1, 3, 2, 2))
    hspoly = HalfspacePolytope(v)

    variables = [v, v]
    indices = np.array([(0, 0, 0, 0), (0, 1, 0, 1)])
    coeffs = np.array([1.0, -1.0])
    b = np.array(10)
    is_open = False
    hspoly.update_constraint(variables, indices, coeffs, b, is_open)

    A, b = hspoly.as_matrix_inequality()
    assert np.allclose(A, np.array([[1.0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0]]))
    assert np.allclose(b, np.array([10]))


def test_as_matrix_inequality_include_bounds_0():
    v = Variable((1, 3, 2, 2))
    hspoly = HalfspacePolytope(v)

    variables = [v, v]
    indices = np.array([(0, 0, 0, 0), (0, 1, 0, 1)])
    coeffs = np.array([1.0, -1.0])
    b = np.array(10)
    is_open = False
    hspoly.update_constraint(variables, indices, coeffs, b, is_open)

    A, b = hspoly.as_matrix_inequality(include_bounds=True)
    assert np.allclose(A, np.array([[1.0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0]]))
    assert np.allclose(b, np.array([10]))


def test_as_matrix_inequality_include_bounds_1():
    v = Variable((1, 3, 2, 2))
    hspoly = HalfspacePolytope(v)

    variables = [v, v]
    indices = np.array([(0, 0, 0, 0), (0, 1, 0, 1)])
    coeffs = np.array([1.0, -1.0])
    b = np.array(10)
    is_open = False
    hspoly.update_constraint(variables, indices, coeffs, b, is_open)

    variables = [v]
    indices = np.array([(0, 0, 0, 0)])
    coeffs = np.array([1.0])
    b = np.array(5)
    is_open = False
    hspoly.update_constraint(variables, indices, coeffs, b, is_open)
    assert hspoly.is_consistent

    variables = [v]
    indices = np.array([(0, 0, 0, 0)])
    coeffs = np.array([-1.0])
    b = np.array(-2)
    is_open = False
    hspoly.update_constraint(variables, indices, coeffs, b, is_open)

    A, b = hspoly.as_matrix_inequality(include_bounds=True)
    assert np.allclose(
        A,
        np.array(
            [
                [1.0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0],
                [-1.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [1.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            ]
        ),
    )
    assert np.allclose(b, np.array([10.0, -2, 5]))


def test_bound_tightening():
    v = Variable((1, 2))
    hspoly = HalfspacePolytope(v)

    # v[0,0] >= 0
    variables = [v]
    indices = np.array([(0, 0)])
    coeffs = np.array([-1.0])
    b = np.array([0.0])
    hspoly.update_constraint(variables, indices, coeffs, b)

    # v[0,1] >= -200
    variables = [v]
    indices = np.array([(0, 1)])
    coeffs = np.array([-1.0])
    b = np.array([200.0])
    hspoly.update_constraint(variables, indices, coeffs, b)

    # v[0,0] <= 100
    variables = [v]
    indices = np.array([(0, 0)])
    coeffs = np.array([1.0])
    b = np.array([100.0])
    hspoly.update_constraint(variables, indices, coeffs, b)

    # v[0,1] <= 100
    variables = [v]
    indices = np.array([(0, 1)])
    coeffs = np.array([1.0])
    b = np.array([100.0])
    hspoly.update_constraint(variables, indices, coeffs, b)

    # v[0,0] - v[0,1] <= 50
    variables = [v, v]
    indices = np.array([(0, 0), (0, 1)])
    coeffs = np.array([1.0, -1.0])
    b = np.array([50.0])
    hspoly.update_constraint(variables, indices, coeffs, b)

    # -v[0,0] + 0.01*v[0,1] <= 1.0
    variables = [v, v]
    indices = np.array([(0, 0), (0, 1)])
    coeffs = np.array([-1.0, 0.01])
    b = np.array([1.0])
    hspoly.update_constraint(variables, indices, coeffs, b)

    assert np.all(hspoly._lower_bound >= np.array([0, -52]))
    assert np.all(hspoly._upper_bound <= np.array([100, 100]))
