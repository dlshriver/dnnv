import numpy as np
import pytest

from dnnv.nn.graph import OperationGraph
from dnnv.nn import operations
from dnnv.properties.expressions import Network
from dnnv.verifiers.common.reductions.iopolytope import *
from dnnv.verifiers.common.reductions.iopolytope import Variable


def setup_function():
    Variable._count = 0


def test_init_merge():
    input_op_0 = operations.Input((1, 5), np.dtype(np.float32))
    add_op_0 = operations.Add(input_op_0, 1)
    op_graph_0 = OperationGraph([add_op_0])
    N0 = Network("N0").concretize(op_graph_0)

    input_op_1 = operations.Input((1, 5), np.dtype(np.float32))
    sub_op_1 = operations.Sub(input_op_1, 1)
    op_graph_1 = OperationGraph([sub_op_1])
    N1 = Network("N1").concretize(op_graph_1)

    input_constraint = HalfspacePolytope()
    output_constraint = HalfspacePolytope()

    prop = IOPolytopeProperty([N0, N1], input_constraint, output_constraint)
    assert len(prop.op_graph.output_operations) == 2
    assert isinstance(prop.op_graph.output_operations[0], operations.Add)
    assert isinstance(prop.op_graph.output_operations[1], operations.Sub)
    assert len(prop.op_graph.input_details) == 1


def test_str():
    input_op = operations.Input((1, 5), np.dtype(np.float32))
    add_op = operations.Add(input_op, 1)
    op_graph = OperationGraph([add_op])
    N = Network("N").concretize(op_graph)

    vi = Variable((1, 5))
    input_constraint = HalfspacePolytope(vi)
    input_constraint.update_constraint([vi], np.array([(0, 1)]), np.array([1.0]), 5.0)
    vo = Variable((1, 5))
    output_constraint = HalfspacePolytope(vo)
    output_constraint.update_constraint([vo], np.array([(0, 0)]), np.array([2.0]), 12.0)

    prop = IOPolytopeProperty([N], input_constraint, output_constraint)
    assert str(prop) == (
        "Property:\n"
        "  Networks:\n"
        "    [Network('N')]\n"
        "  Input Constraint:\n"
        "    1.0 * x_0[(0, 1)] <= 5.0\n"
        "  Output Constraint:\n"
        "    2.0 * x_1[(0, 0)] <= 12.0"
    )


def test_validate_counter_example_true():
    input_op = operations.Input((1, 2), np.dtype(np.float32))
    matmul_op = operations.MatMul(input_op, np.array([[1.0], [1.0]], dtype=np.float32))
    add_op = operations.Add(matmul_op, 1)
    op_graph = OperationGraph([add_op])
    N = Network("N").concretize(op_graph)

    vi = Variable((1, 2))
    input_constraint = HalfspacePolytope(vi)
    variables = [vi, vi]
    indices = np.array([(0, 0), (0, 1)])
    coeffs = np.array([1.0, 1.0])
    b = np.array(2)
    input_constraint.update_constraint(variables, indices, coeffs, b)
    coeffs = np.array([-1.0, -1.0])
    b = np.array(2)
    input_constraint.update_constraint(variables, indices, coeffs, b)
    coeffs = np.array([1.0, -1.0])
    b = np.array(4)
    input_constraint.update_constraint(variables, indices, coeffs, b)
    coeffs = np.array([-1.0, 1.0])
    b = np.array(11)
    input_constraint.update_constraint(variables, indices, coeffs, b)

    vo = Variable((1, 1))
    output_constraint = HalfspacePolytope(vo)
    variables = [vo]
    indices = np.array([(0, 0)])
    coeffs = np.array([1.0])
    b = np.array(2)
    output_constraint.update_constraint(variables, indices, coeffs, b)
    coeffs = np.array([-1.0])
    b = np.array(1)
    output_constraint.update_constraint(variables, indices, coeffs, b)

    prop = IOPolytopeProperty([N], input_constraint, output_constraint)
    x = np.array([[0.0, 0.0]]).astype(np.float32)
    assert prop.validate_counter_example(x)[0]
    x = np.array([[0.5, 0.5]]).astype(np.float32)
    assert prop.validate_counter_example(x)[0]
    x = np.array([[-1.0, 0.0]]).astype(np.float32)
    assert prop.validate_counter_example(x)[0]


def test_validate_counter_example_false():
    input_op = operations.Input((1, 2), np.dtype(np.float32))
    matmul_op = operations.MatMul(input_op, np.array([[1.0], [1.0]], dtype=np.float32))
    add_op = operations.Add(matmul_op, 1)
    op_graph = OperationGraph([add_op])
    N = Network("N").concretize(op_graph)

    vi = Variable((1, 2))
    input_constraint = HalfspacePolytope(vi)
    variables = [vi, vi]
    indices = np.array([(0, 0), (0, 1)])
    coeffs = np.array([1.0, 1.0])
    b = np.array(2)
    input_constraint.update_constraint(variables, indices, coeffs, b)
    coeffs = np.array([-1.0, -1.0])
    b = np.array(2)
    input_constraint.update_constraint(variables, indices, coeffs, b)
    coeffs = np.array([1.0, -1.0])
    b = np.array(4)
    input_constraint.update_constraint(variables, indices, coeffs, b)
    coeffs = np.array([-1.0, 1.0])
    b = np.array(11)
    input_constraint.update_constraint(variables, indices, coeffs, b)

    vo = Variable((1, 1))
    output_constraint = HalfspacePolytope(vo)
    variables = [vo]
    indices = np.array([(0, 0)])
    coeffs = np.array([1.0])
    b = np.array(1)
    output_constraint.update_constraint(variables, indices, coeffs, b)
    coeffs = np.array([-1.0])
    b = np.array(1)
    output_constraint.update_constraint(variables, indices, coeffs, b)

    prop = IOPolytopeProperty([N], input_constraint, output_constraint)
    x = np.array([[0.0, 110.0]]).astype(np.float32)
    assert not prop.validate_counter_example(x)[0]
    x = np.array([[1.0, 0.5]]).astype(np.float32)
    assert not prop.validate_counter_example(x)[0]


def test_suffixed_op_graph():
    input_op = operations.Input((1, 2), np.dtype(np.float32))
    matmul_op = operations.MatMul(input_op, np.array([[1.0], [1.0]], dtype=np.float32))
    add_op = operations.Add(matmul_op, 1)
    relu_op = operations.Relu(add_op)
    op_graph = OperationGraph([relu_op])
    N = Network("N").concretize(op_graph)

    vi = Variable((1, 2))
    input_constraint = HalfspacePolytope(vi)
    variables = [vi, vi]
    indices = np.array([(0, 0), (0, 1)])
    coeffs = np.array([1.0, 1.0])
    b = np.array(2)
    input_constraint.update_constraint(variables, indices, coeffs, b)
    coeffs = np.array([-1.0, -1.0])
    b = np.array(2)
    input_constraint.update_constraint(variables, indices, coeffs, b)
    coeffs = np.array([1.0, -1.0])
    b = np.array(4)
    input_constraint.update_constraint(variables, indices, coeffs, b)
    coeffs = np.array([-1.0, 1.0])
    b = np.array(11)
    input_constraint.update_constraint(variables, indices, coeffs, b)

    vo = Variable((1, 1))
    output_constraint = HalfspacePolytope(vo)
    variables = [vo]
    indices = np.array([(0, 0)])
    coeffs = np.array([1.0])
    b = np.array(1)
    output_constraint.update_constraint(variables, indices, coeffs, b)
    coeffs = np.array([-1.0])
    b = np.array(1)
    output_constraint.update_constraint(variables, indices, coeffs, b)

    prop = IOPolytopeProperty([N], input_constraint, output_constraint)
    suffixed_op_graph = prop.suffixed_op_graph()

    x = np.array([[1.0, 0.5]]).astype(np.float32)
    assert suffixed_op_graph(x).item() > 0

    x = np.array([[0.0, 0.0]]).astype(np.float32)
    assert suffixed_op_graph(x).item() <= 0
    x = np.array([[0.25, -0.25]]).astype(np.float32)
    assert suffixed_op_graph(x).item() <= 0
    x = np.array([[-1.0, 0.0]]).astype(np.float32)
    assert suffixed_op_graph(x).item() <= 0


def test_suffixed_op_graph_multiple_output_ops():
    input_op = operations.Input((1, 2), np.dtype(np.float32))
    matmul_op = operations.MatMul(input_op, np.array([[1.0], [1.0]], dtype=np.float32))
    add_op = operations.Add(matmul_op, 1)
    relu_op = operations.Relu(add_op)
    tanh_op = operations.Tanh(add_op)
    op_graph = OperationGraph([relu_op, tanh_op])
    N = Network("N").concretize(op_graph)

    vi = Variable((1, 2))
    input_constraint = HalfspacePolytope(vi)
    variables = [vi, vi]
    indices = np.array([(0, 0), (0, 1)])
    coeffs = np.array([1.0, 1.0])
    b = np.array(2)
    input_constraint.update_constraint(variables, indices, coeffs, b)
    coeffs = np.array([-1.0, -1.0])
    b = np.array(2)
    input_constraint.update_constraint(variables, indices, coeffs, b)
    coeffs = np.array([1.0, -1.0])
    b = np.array(4)
    input_constraint.update_constraint(variables, indices, coeffs, b)
    coeffs = np.array([-1.0, 1.0])
    b = np.array(11)
    input_constraint.update_constraint(variables, indices, coeffs, b)

    vo1 = Variable((1, 1))
    vo2 = Variable((1, 1))
    output_constraint = HalfspacePolytope(vo1)
    output_constraint.add_variable(vo2)
    variables = [vo1]
    indices = np.array([(0, 0)])
    coeffs = np.array([1.0])
    b = np.array(1)
    output_constraint.update_constraint(variables, indices, coeffs, b)
    coeffs = np.array([-1.0])
    b = np.array(1)
    output_constraint.update_constraint(variables, indices, coeffs, b)
    variables = [vo2]
    indices = np.array([(0, 0)])
    coeffs = np.array([0.5])
    b = np.array(1)
    output_constraint.update_constraint(variables, indices, coeffs, b, is_open=True)
    coeffs = np.array([-0.5])
    b = np.array(1)
    output_constraint.update_constraint(variables, indices, coeffs, b, is_open=True)

    prop = IOPolytopeProperty([N], input_constraint, output_constraint)
    suffixed_op_graph = prop.suffixed_op_graph()

    x = np.array([[1.0, 0.5]]).astype(np.float32)
    assert suffixed_op_graph(x).item() > 0
    x = np.array([[2.0, 1.0]]).astype(np.float32)
    assert prop.validate_counter_example(x)
    assert suffixed_op_graph(x).item() > 0

    x = np.array([[0.0, 0.0]]).astype(np.float32)
    assert suffixed_op_graph(x).item() <= 0
    x = np.array([[0.25, -0.25]]).astype(np.float32)
    assert suffixed_op_graph(x).item() <= 0
    x = np.array([[-1.0, 0.0]]).astype(np.float32)
    assert suffixed_op_graph(x).item() <= 0


def test_prefixed_and_suffixed_op_graph_hspoly_input_constraints():
    input_op = operations.Input((1, 2), np.dtype(np.float32))
    matmul_op = operations.MatMul(input_op, np.array([[1.0], [1.0]], dtype=np.float32))
    add_op = operations.Add(matmul_op, 1)
    relu_op = operations.Relu(add_op)
    op_graph = OperationGraph([relu_op])
    N = Network("N").concretize(op_graph)

    vi = Variable((1, 2))
    input_constraint = HalfspacePolytope(vi)
    variables = [vi]
    indices = np.array([(0, 0)])
    coeffs = np.array([1.0])
    b = np.array(2)
    input_constraint.update_constraint(variables, indices, coeffs, b)
    input_constraint.update_constraint(variables, indices, -coeffs, b)
    indices = np.array([(0, 1)])
    input_constraint.update_constraint(variables, indices, coeffs, b)
    input_constraint.update_constraint(variables, indices, -coeffs, b)

    vo = Variable((1, 1))
    output_constraint = HalfspacePolytope(vo)
    variables = [vo]
    indices = np.array([(0, 0)])
    coeffs = np.array([1.0])
    b = np.array(1)
    output_constraint.update_constraint(variables, indices, coeffs, b)
    coeffs = np.array([-1.0])
    b = np.array(1)
    output_constraint.update_constraint(variables, indices, coeffs, b)

    prop = IOPolytopeProperty([N], input_constraint, output_constraint)
    with pytest.raises(
        ValueError,
        match="HalfspacePolytope input constraints are not yet supported",
    ):
        _ = prop.prefixed_and_suffixed_op_graph()


def test_prefixed_and_suffixed_op_graph():
    input_op = operations.Input((1, 2), np.dtype(np.float32))
    matmul_op = operations.MatMul(input_op, np.array([[1.0], [1.0]], dtype=np.float32))
    add_op = operations.Add(matmul_op, 1)
    relu_op = operations.Relu(add_op)
    op_graph = OperationGraph([relu_op])
    N = Network("N").concretize(op_graph)

    vi = Variable((1, 2))
    input_constraint = HyperRectangle(vi)
    variables = [vi]
    indices = np.array([(0, 0)])
    coeffs = np.array([1.0])
    b = np.array(2)
    input_constraint.update_constraint(variables, indices, coeffs, b)
    input_constraint.update_constraint(variables, indices, -coeffs, b)
    indices = np.array([(0, 1)])
    input_constraint.update_constraint(variables, indices, coeffs, b)
    input_constraint.update_constraint(variables, indices, -coeffs, b)

    vo = Variable((1, 1))
    output_constraint = HalfspacePolytope(vo)
    variables = [vo]
    indices = np.array([(0, 0)])
    coeffs = np.array([1.0])
    b = np.array(1)
    output_constraint.update_constraint(variables, indices, coeffs, b)
    coeffs = np.array([-1.0])
    b = np.array(1)
    output_constraint.update_constraint(variables, indices, coeffs, b)

    prop = IOPolytopeProperty([N], input_constraint, output_constraint)
    prefixed_and_suffixed_op_graph, (lbs, ubs) = prop.prefixed_and_suffixed_op_graph()

    x = np.array([[1.0, 0.5]]).astype(np.float32)
    assert prefixed_and_suffixed_op_graph(x).item() > 0
    x = np.array([[1.0, 1.0]]).astype(np.float32)
    assert prefixed_and_suffixed_op_graph(x).item() > 0

    x = np.array([[0.0, 0.0]]).astype(np.float32)
    assert prefixed_and_suffixed_op_graph(x).item() <= 0
    x = np.array([[0.5, 0.5]]).astype(np.float32)
    assert prefixed_and_suffixed_op_graph(x).item() <= 0
    x = np.array([[-1.0, 0.0]]).astype(np.float32)
    assert prefixed_and_suffixed_op_graph(x).item() <= 0


def test_prefixed_and_suffixed_op_graph_4d_input():
    input_op = operations.Input((1, 2, 1, 1), np.dtype(np.float32))
    conv_op = operations.Conv(
        input_op,
        np.ones((1, 2, 1, 1), dtype=np.float32),
        np.ones((1,), dtype=np.float32),
    )
    relu_op = operations.Relu(conv_op)
    op_graph = OperationGraph([relu_op])
    N = Network("N").concretize(op_graph)

    vi = Variable((1, 2, 1, 1))
    input_constraint = HyperRectangle(vi)
    variables = [vi]
    indices = np.array([(0, 0, 0, 0)])
    coeffs = np.array([1.0])
    b = np.array(2)
    input_constraint.update_constraint(variables, indices, coeffs, b)
    input_constraint.update_constraint(variables, indices, -coeffs, b)
    indices = np.array([(0, 1, 0, 0)])
    input_constraint.update_constraint(variables, indices, coeffs, b)
    input_constraint.update_constraint(variables, indices, -coeffs, b)

    vo = Variable((1, 1))
    output_constraint = HalfspacePolytope(vo)
    variables = [vo]
    indices = np.array([(0, 0)])
    coeffs = np.array([1.0])
    b = np.array(1)
    output_constraint.update_constraint(variables, indices, coeffs, b)
    coeffs = np.array([-1.0])
    b = np.array(1)
    output_constraint.update_constraint(variables, indices, coeffs, b)

    prop = IOPolytopeProperty([N], input_constraint, output_constraint)
    prefixed_and_suffixed_op_graph, (lbs, ubs) = prop.prefixed_and_suffixed_op_graph()

    x = np.array([[[[1.0]], [[0.5]]]]).astype(np.float32)
    assert prefixed_and_suffixed_op_graph(x).item() > 0
    x = np.array([[[[1.0]], [[1.0]]]]).astype(np.float32)
    assert prefixed_and_suffixed_op_graph(x).item() > 0

    x = np.array([[[[0.0]], [[0.0]]]]).astype(np.float32)
    assert prefixed_and_suffixed_op_graph(x).item() <= 0
    x = np.array([[[[0.5]], [[0.5]]]]).astype(np.float32)
    assert prefixed_and_suffixed_op_graph(x).item() <= 0
    x = np.array([[[[-1.0]], [[0.0]]]]).astype(np.float32)
    assert prefixed_and_suffixed_op_graph(x).item() <= 0
