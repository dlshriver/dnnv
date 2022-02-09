import numpy as np
import pytest

from dnnv.nn.graph import OperationGraph
from dnnv.nn import operations
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


# TODO : finish this
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


def test_bounds_monotonous():
    """
    If we initiate the bounds for a certain upper and lower bound, the lower/upper bound may only
    increase/decrease.
    """
    reduction = IOPolytopeReduction(HalfspacePolytope)

    # Example with bounds initialized to [0, -200] [100,100]
    phi = Exists(
        Symbol("x"),
        And(
            (
                (
                    Constant(-0.014999999999981656)
                    * Subscript(Symbol("x"), Constant((0, 1)))
                )
                < Constant(1.4999999999926625)
            ),
            And(
                (
                    (
                        (
                            Constant(-0.007499999999983742)
                            * Subscript(Symbol("x"), Constant((0, 1)))
                        )
                    )
                    < Constant(0.4999999999967484)
                ),
                And(
                    (
                        (
                            (
                                Constant(0.0075000000000270415)
                                * Subscript(Symbol("x"), Constant((0, 1)))
                            )
                        )
                        <= Constant(0.500000000000365)
                    ),
                    And(
                        (
                            (
                                (
                                    Constant(-0.007499999999983742)
                                    * Subscript(Symbol("x"), Constant((0, 1)))
                                )
                            )
                            <= Constant(1.4999999999967484)
                        ),
                        And(
                            (
                                (
                                    (1 * Subscript(Symbol("x"), Constant((0, 0))))
                                    + (
                                        Constant(-0.13574999999783516)
                                        * Subscript(Symbol("x"), Constant((0, 1)))
                                    )
                                )
                                < Constant(47.64999999956703)
                            ),
                            And(
                                (((1 * Network("N")(Symbol("x")))) < Constant(100.0)),
                                And(
                                    (((-1 * Symbol("x"))) <= np.array([[0, 200]])),
                                    And(
                                        (
                                            (
                                                (
                                                    Constant(0.015000000000010782)
                                                    * Subscript(
                                                        Symbol("x"), Constant((0, 1))
                                                    )
                                                )
                                            )
                                            <= Constant(-0.9999999999963833)
                                        ),
                                        And(
                                            (
                                                ((1 * Symbol("x")))
                                                <= np.array([[100, 100]])
                                            ),
                                            And(
                                                (
                                                    (
                                                        (
                                                            -1
                                                            * Subscript(
                                                                Symbol("x"),
                                                                Constant((0, 0)),
                                                            )
                                                        )
                                                        + (
                                                            Constant(
                                                                0.010000000000036688
                                                            )
                                                            * Subscript(
                                                                Symbol("x"),
                                                                Constant((0, 1)),
                                                            )
                                                        )
                                                    )
                                                    <= Constant(1.0)
                                                ),
                                                And(
                                                    (
                                                        (
                                                            (
                                                                1
                                                                * Subscript(
                                                                    Symbol("x"),
                                                                    Constant((0, 1)),
                                                                )
                                                            )
                                                        )
                                                        < 0
                                                    ),
                                                    And(
                                                        (
                                                            (
                                                                (
                                                                    Constant(
                                                                        -0.014999999999981656
                                                                    )
                                                                    * Subscript(
                                                                        Symbol("x"),
                                                                        Constant(
                                                                            (0, 1)
                                                                        ),
                                                                    )
                                                                )
                                                            )
                                                            < Constant(
                                                                1.4999999999926625
                                                            )
                                                        ),
                                                        And(
                                                            (
                                                                (
                                                                    (
                                                                        Constant(
                                                                            0.005000000000018344
                                                                        )
                                                                        * Subscript(
                                                                            Symbol("x"),
                                                                            Constant(
                                                                                (0, 1)
                                                                            ),
                                                                        )
                                                                    )
                                                                )
                                                                <= Constant(0.5)
                                                            ),
                                                            (
                                                                (
                                                                    (
                                                                        Constant(
                                                                            -0.009999999999963313
                                                                        )
                                                                        * Subscript(
                                                                            Symbol("x"),
                                                                            Constant(
                                                                                (0, 1)
                                                                            ),
                                                                        )
                                                                    )
                                                                )
                                                                < Constant(
                                                                    0.9999999999926624
                                                                )
                                                            ),
                                                        ),
                                                    ),
                                                ),
                                            ),
                                        ),
                                    ),
                                ),
                            ),
                        ),
                    ),
                ),
            ),
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
