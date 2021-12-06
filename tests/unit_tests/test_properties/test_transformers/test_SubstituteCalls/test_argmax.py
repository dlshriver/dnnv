import numpy as np
import pytest

from dnnv.properties.expressions import *
from dnnv.properties.transformers import SubstituteCalls


def test_argmax_symbol():
    expr = Constant(np.argmax)(Symbol("x"))
    new_expr = SubstituteCalls().visit(expr)
    assert new_expr is not expr
    assert new_expr.is_equivalent(expr)


def test_argmax_non_concrete_network():
    expr = Constant(np.argmax)(Network("N")(Symbol("x")))
    new_expr = SubstituteCalls().visit(expr)
    assert new_expr is not expr
    assert new_expr.is_equivalent(expr)


def test_argmax_constant():
    expr = Constant(np.argmax)(Constant(np.array([1, 2, 5, 3, 4])))
    new_expr = SubstituteCalls().visit(expr)
    assert new_expr is not expr
    assert new_expr is Constant(2)


def test_argmax_concrete_network():
    fake_network = lambda x: x
    fake_network.output_shape = [(1, 3)]

    expr = Constant(np.argmax)(Network("N")(Symbol("x")))
    expr.concretize(N=fake_network)

    new_expr = SubstituteCalls().visit(expr)
    expected_expr = IfThenElse(
        And(
            GreaterThanOrEqual(
                Network("N")(Symbol("x"))[(0, 0)], Network("N")(Symbol("x"))[(0, 1)]
            ),
            GreaterThanOrEqual(
                Network("N")(Symbol("x"))[(0, 0)], Network("N")(Symbol("x"))[(0, 2)]
            ),
        ),
        Constant(0),
        IfThenElse(
            And(
                GreaterThanOrEqual(
                    Network("N")(Symbol("x"))[(0, 1)], Network("N")(Symbol("x"))[(0, 2)]
                )
            ),
            Constant(1),
            Constant(2),
        ),
    )
    assert new_expr is not expr
    assert new_expr.is_equivalent(expected_expr)


def test_argmax_equal_too_many_args():
    expr = Constant(np.argmax)(Symbol("x"), Symbol("a")) == Symbol("y")
    with pytest.raises(RuntimeError, match="Too many arguments for argcmp"):
        _ = SubstituteCalls().visit(expr)


def test_argmax_symbol_equal_symbol():
    expr = Constant(np.argmax)(Symbol("x")) == Symbol("y")
    new_expr = SubstituteCalls().visit(expr)
    assert new_expr is not expr
    assert new_expr.is_equivalent(expr)


def test_argmax_symbol_equal_constant():
    expr = Constant(np.argmax)(Symbol("x")) == Constant(0)
    new_expr = SubstituteCalls().visit(expr)
    assert new_expr is not expr
    assert new_expr.is_equivalent(expr)


def test_argmax_constant_equal_constant():
    expr = Constant(np.argmax)(Constant(np.array([2, 1, 5, 3, 4]))) == Constant(0)
    new_expr = SubstituteCalls().visit(expr)
    assert new_expr is not expr
    assert new_expr is Constant(False)

    expr = Constant(np.argmax)(Constant(np.array([2, 1, 5, 3, 4]))) == Constant(2)
    new_expr = SubstituteCalls().visit(expr)
    assert new_expr is not expr
    assert new_expr is Constant(True)

    expr = Constant(4) == Constant(np.argmax)(Constant(np.array([2, 1, 5, 3, 4])))
    new_expr = SubstituteCalls().visit(expr)
    assert new_expr is not expr
    assert new_expr is Constant(False)

    expr = Constant(2) == Constant(np.argmax)(Constant(np.array([2, 1, 5, 3, 4])))
    new_expr = SubstituteCalls().visit(expr)
    assert new_expr is not expr
    assert new_expr is Constant(True)


def test_argmax_concrete_network_equal_constant():
    fake_network = lambda x: x
    fake_network.output_shape = [(1, 3)]

    expr = Constant(np.argmax)(Network("N")(Symbol("x"))) == Constant(0)
    expr.concretize(N=fake_network)

    new_expr = SubstituteCalls().visit(expr)
    expected_expr = And(
        GreaterThanOrEqual(
            Network("N")(Symbol("x"))[(0, 0)], Network("N")(Symbol("x"))[(0, 1)]
        ),
        GreaterThanOrEqual(
            Network("N")(Symbol("x"))[(0, 0)], Network("N")(Symbol("x"))[(0, 2)]
        ),
    )
    assert new_expr is not expr
    assert new_expr.is_equivalent(expected_expr)


def test_argmax_concrete_network_equal_symbol():
    fake_network = lambda x: x
    fake_network.output_shape = [(1, 3)]

    expr = Constant(np.argmax)(Network("N")(Symbol("x"))) == Symbol("y")
    expr.concretize(N=fake_network)

    new_expr = SubstituteCalls().visit(expr)
    expected_expr = And(
        Implies(
            And(
                GreaterThanOrEqual(
                    Network("N")(Symbol("x"))[(0, 0)], Network("N")(Symbol("x"))[(0, 1)]
                ),
                GreaterThanOrEqual(
                    Network("N")(Symbol("x"))[(0, 0)], Network("N")(Symbol("x"))[(0, 2)]
                ),
            ),
            Equal(Symbol("y"), Constant(0)),
        ),
        Implies(
            And(
                GreaterThanOrEqual(
                    Network("N")(Symbol("x"))[(0, 1)], Network("N")(Symbol("x"))[(0, 0)]
                ),
                GreaterThanOrEqual(
                    Network("N")(Symbol("x"))[(0, 1)], Network("N")(Symbol("x"))[(0, 2)]
                ),
            ),
            Equal(Symbol("y"), Constant(1)),
        ),
        Implies(
            And(
                GreaterThanOrEqual(
                    Network("N")(Symbol("x"))[(0, 2)], Network("N")(Symbol("x"))[(0, 0)]
                ),
                GreaterThanOrEqual(
                    Network("N")(Symbol("x"))[(0, 2)], Network("N")(Symbol("x"))[(0, 1)]
                ),
            ),
            Equal(Symbol("y"), Constant(2)),
        ),
    )
    assert new_expr is not expr
    assert new_expr.is_equivalent(expected_expr)


def test_argmax_symbol_notequal_symbol():
    expr = Constant(np.argmax)(Symbol("x")) != Symbol("y")
    new_expr = SubstituteCalls().visit(expr)
    assert new_expr is not expr
    assert new_expr.is_equivalent(expr)


def test_argmax_symbol_notequal_constant():
    expr = Constant(np.argmax)(Symbol("x")) != Constant(0)
    new_expr = SubstituteCalls().visit(expr)
    assert new_expr is not expr
    assert new_expr.is_equivalent(expr)


def test_argmax_constant_notequal_constant():
    expr = Constant(np.argmax)(Constant(np.array([2, 1, 5, 3, 4]))) != Constant(0)
    new_expr = SubstituteCalls().visit(expr).propagate_constants()
    assert new_expr is not expr
    assert new_expr is Constant(True)

    expr = Constant(np.argmax)(Constant(np.array([2, 1, 5, 3, 4]))) != Constant(2)
    new_expr = SubstituteCalls().visit(expr).propagate_constants()
    assert new_expr is not expr
    assert new_expr is Constant(False)


def test_argmax_concrete_network_notequal_constant():
    fake_network = lambda x: x
    fake_network.output_shape = [(1, 3)]

    expr = Constant(np.argmax)(Network("N")(Symbol("x"))) != Constant(0)
    expr.concretize(N=fake_network)

    new_expr = SubstituteCalls().visit(expr)
    expected_expr = Or(
        LessThan(Network("N")(Symbol("x"))[(0, 0)], Network("N")(Symbol("x"))[(0, 1)]),
        LessThan(Network("N")(Symbol("x"))[(0, 0)], Network("N")(Symbol("x"))[(0, 2)]),
    )
    assert new_expr is not expr
    assert new_expr.is_equivalent(expected_expr)


def test_argmax_concrete_network_notequal_symbol():
    fake_network = lambda x: x
    fake_network.output_shape = [(1, 3)]

    expr = Constant(np.argmax)(Network("N")(Symbol("x"))) != Symbol("y")
    expr.concretize(N=fake_network)

    new_expr = SubstituteCalls().visit(expr)
    expected_expr = Or(
        And(
            GreaterThanOrEqual(
                Network("N")(Symbol("x"))[(0, 0)], Network("N")(Symbol("x"))[(0, 1)]
            ),
            GreaterThanOrEqual(
                Network("N")(Symbol("x"))[(0, 0)], Network("N")(Symbol("x"))[(0, 2)]
            ),
            NotEqual(Symbol("y"), Constant(0)),
        ),
        And(
            GreaterThanOrEqual(
                Network("N")(Symbol("x"))[(0, 1)], Network("N")(Symbol("x"))[(0, 0)]
            ),
            GreaterThanOrEqual(
                Network("N")(Symbol("x"))[(0, 1)], Network("N")(Symbol("x"))[(0, 2)]
            ),
            NotEqual(Symbol("y"), Constant(1)),
        ),
        And(
            GreaterThanOrEqual(
                Network("N")(Symbol("x"))[(0, 2)], Network("N")(Symbol("x"))[(0, 0)]
            ),
            GreaterThanOrEqual(
                Network("N")(Symbol("x"))[(0, 2)], Network("N")(Symbol("x"))[(0, 1)]
            ),
            NotEqual(Symbol("y"), Constant(2)),
        ),
    )
    assert new_expr is not expr
    assert new_expr.is_equivalent(expected_expr)
