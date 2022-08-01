import numpy as np
import pytest

from dnnv.nn import operations
from dnnv.nn.graph import OperationGraph
from dnnv.properties.expressions import *
from dnnv.verifiers.common.reductions.iopolytope import *


def test_non_existential():
    reduction = IOPolytopeReduction()

    phi = Constant(True)
    with pytest.raises(NotImplementedError):
        properties = list(reduction.reduce_property(phi))


def test_no_network():
    reduction = IOPolytopeReduction()

    phi = Exists(Symbol("x"), Symbol("x") > Constant(0))
    with pytest.raises(IOPolytopeReductionError):
        properties = list(reduction.reduce_property(phi))


def test_non_concrete_network():
    reduction = IOPolytopeReduction()

    phi = Exists(Symbol("x"), Network("N")(Symbol("x")) > Constant(0))
    with pytest.raises(IOPolytopeReductionError):
        properties = list(reduction.reduce_property(phi))


def test_simple_property():
    reduction = IOPolytopeReduction()

    phi = Exists(
        Symbol("x"),
        And(
            Constant(0) <= Symbol("x"),
            Symbol("x") <= Constant(1),
            Network("N")(Symbol("x")) > Constant(0),
        ),
    )
    input_op = operations.Input((1,), np.dtype(np.float64))
    output_op = operations.Add(input_op, operations.Mul(np.float64(-2), input_op))
    op_graph = OperationGraph([output_op])
    phi.concretize(N=op_graph)

    properties = list(reduction.reduce_property(phi))
    assert len(properties) == 1
    prop = properties[0]

    assert len(prop.networks) == 1

    assert np.all(prop.input_constraint._lower_bound == np.array([0]))
    assert np.all(prop.input_constraint._upper_bound == np.array([1]))

    assert np.all(prop.output_constraint._lower_bound == np.array([np.nextafter(0, 1)]))
    assert np.all(prop.output_constraint._upper_bound == np.array([np.inf]))


def test_bound_tightening():
    reduction = IOPolytopeReduction(HalfspacePolytope)

    N = Network("N")
    x = Symbol("x")
    phi = Exists(
        x,
        And(
            And(N(x) < Constant(1), Constant(np.array([[0, -200]])) < x),
            And(N(x) > Constant(-1), x < Constant(100)),
            1 * x[0, 0] + -1 * x[0, 1] < 50.0,
            -1 * x[0, 0] + 0.01 * x[0, 1] <= 1.0,
        ),
    )
    input_op = operations.Input((1, 2), np.dtype(np.float64))
    output_op = operations.Add(input_op, operations.Mul(np.float64(-2), input_op))
    op_graph = OperationGraph([output_op])
    phi.concretize(N=op_graph)

    properties = list(reduction.reduce_property(phi))
    assert len(properties) == 1
    prop = properties[0]

    assert len(prop.networks) == 1

    assert np.all(prop.input_constraint._lower_bound >= np.array([0, -200]))
    assert np.all(prop.input_constraint._upper_bound <= np.array([100, 100]))
