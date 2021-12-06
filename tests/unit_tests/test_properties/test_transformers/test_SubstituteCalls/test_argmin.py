import numpy as np
import pytest

from dnnv.properties.expressions import *
from dnnv.properties.transformers import SubstituteCalls


def test_argmin_symbol():
    expr = Constant(np.argmin)(Symbol("x"))
    new_expr = SubstituteCalls().visit(expr)
    assert new_expr is not expr
    assert new_expr.is_equivalent(expr)


def test_argmin_non_concrete_network():
    expr = Constant(np.argmin)(Network("N")(Symbol("x")))
    new_expr = SubstituteCalls().visit(expr)
    assert new_expr is not expr
    assert new_expr.is_equivalent(expr)


def test_argmin_constant():
    expr = Constant(np.argmin)(Constant(np.array([3, 2, 5, 1, 4])))
    new_expr = SubstituteCalls().visit(expr)
    assert new_expr is not expr
    assert new_expr is Constant(3)


def test_argmin_concrete_network():
    fake_network = lambda x: x
    fake_network.output_shape = [(1, 3)]

    expr = Constant(np.argmin)(Network("N")(Symbol("x")))
    expr.concretize(N=fake_network)

    new_expr = SubstituteCalls().visit(expr)
    expected_expr = IfThenElse(
        And(
            LessThanOrEqual(
                Network("N")(Symbol("x"))[(0, 0)], Network("N")(Symbol("x"))[(0, 1)]
            ),
            LessThanOrEqual(
                Network("N")(Symbol("x"))[(0, 0)], Network("N")(Symbol("x"))[(0, 2)]
            ),
        ),
        Constant(0),
        IfThenElse(
            And(
                LessThanOrEqual(
                    Network("N")(Symbol("x"))[(0, 1)], Network("N")(Symbol("x"))[(0, 2)]
                )
            ),
            Constant(1),
            Constant(2),
        ),
    )
    assert new_expr is not expr
    assert new_expr.is_equivalent(expected_expr)


def test_argmin_equal_too_many_args():
    expr = Constant(np.argmin)(Symbol("x"), Symbol("a")) == Symbol("y")
    with pytest.raises(RuntimeError, match="Too many arguments for argcmp"):
        _ = SubstituteCalls().visit(expr)


def test_argmin_symbol_equal_symbol():
    expr = Constant(np.argmin)(Symbol("x")) == Symbol("y")
    new_expr = SubstituteCalls().visit(expr)
    assert new_expr is not expr
    assert new_expr.is_equivalent(expr)


def test_argmin_symbol_equal_constant():
    expr = Constant(np.argmin)(Symbol("x")) == Constant(0)
    new_expr = SubstituteCalls().visit(expr)
    assert new_expr is not expr
    assert new_expr.is_equivalent(expr)


def test_argmin_constant_equal_constant():
    expr = Constant(np.argmin)(Constant(np.array([2, 1, 5, 3, 4]))) == Constant(0)
    new_expr = SubstituteCalls().visit(expr)
    assert new_expr is not expr
    assert new_expr is Constant(False)

    expr = Constant(np.argmin)(Constant(np.array([2, 1, 5, 3, 4]))) == Constant(1)
    new_expr = SubstituteCalls().visit(expr)
    assert new_expr is not expr
    assert new_expr is Constant(True)

    expr = Constant(4) == Constant(np.argmin)(Constant(np.array([2, 1, 5, 3, 4])))
    new_expr = SubstituteCalls().visit(expr)
    assert new_expr is not expr
    assert new_expr is Constant(False)

    expr = Constant(1) == Constant(np.argmin)(Constant(np.array([2, 1, 5, 3, 4])))
    new_expr = SubstituteCalls().visit(expr)
    assert new_expr is not expr
    assert new_expr is Constant(True)


def test_argmin_concrete_network_equal_constant():
    fake_network = lambda x: x
    fake_network.output_shape = [(1, 3)]

    expr = Constant(np.argmin)(Network("N")(Symbol("x"))) == Constant(0)
    expr.concretize(N=fake_network)

    new_expr = SubstituteCalls().visit(expr)
    expected_expr = And(
        LessThanOrEqual(
            Network("N")(Symbol("x"))[(0, 0)], Network("N")(Symbol("x"))[(0, 1)]
        ),
        LessThanOrEqual(
            Network("N")(Symbol("x"))[(0, 0)], Network("N")(Symbol("x"))[(0, 2)]
        ),
    )
    assert new_expr is not expr
    assert new_expr.is_equivalent(expected_expr)


def test_argmin_concrete_network_equal_symbol():
    fake_network = lambda x: x
    fake_network.output_shape = [(1, 3)]

    expr = Constant(np.argmin)(Network("N")(Symbol("x"))) == Symbol("y")
    expr.concretize(N=fake_network)

    new_expr = SubstituteCalls().visit(expr)
    expected_expr = And(
        Implies(
            And(
                LessThanOrEqual(
                    Network("N")(Symbol("x"))[(0, 0)], Network("N")(Symbol("x"))[(0, 1)]
                ),
                LessThanOrEqual(
                    Network("N")(Symbol("x"))[(0, 0)], Network("N")(Symbol("x"))[(0, 2)]
                ),
            ),
            Equal(Symbol("y"), Constant(0)),
        ),
        Implies(
            And(
                LessThanOrEqual(
                    Network("N")(Symbol("x"))[(0, 1)], Network("N")(Symbol("x"))[(0, 0)]
                ),
                LessThanOrEqual(
                    Network("N")(Symbol("x"))[(0, 1)], Network("N")(Symbol("x"))[(0, 2)]
                ),
            ),
            Equal(Symbol("y"), Constant(1)),
        ),
        Implies(
            And(
                LessThanOrEqual(
                    Network("N")(Symbol("x"))[(0, 2)], Network("N")(Symbol("x"))[(0, 0)]
                ),
                LessThanOrEqual(
                    Network("N")(Symbol("x"))[(0, 2)], Network("N")(Symbol("x"))[(0, 1)]
                ),
            ),
            Equal(Symbol("y"), Constant(2)),
        ),
    )
    assert new_expr is not expr
    assert new_expr.is_equivalent(expected_expr)


def test_argmin_symbol_notequal_symbol():
    expr = Constant(np.argmin)(Symbol("x")) != Symbol("y")
    new_expr = SubstituteCalls().visit(expr)
    assert new_expr is not expr
    assert new_expr.is_equivalent(expr)


def test_argmin_symbol_notequal_constant():
    expr = Constant(np.argmin)(Symbol("x")) != Constant(0)
    new_expr = SubstituteCalls().visit(expr)
    assert new_expr is not expr
    assert new_expr.is_equivalent(expr)


def test_argmin_constant_notequal_constant():
    expr = Constant(np.argmin)(Constant(np.array([2, 1, 5, 3, 4]))) != Constant(0)
    new_expr = SubstituteCalls().visit(expr).propagate_constants()
    assert new_expr is not expr
    assert new_expr is Constant(True)

    expr = Constant(np.argmin)(Constant(np.array([2, 1, 5, 3, 4]))) != Constant(1)
    new_expr = SubstituteCalls().visit(expr).propagate_constants()
    assert new_expr is not expr
    assert new_expr is Constant(False)


def test_argmin_concrete_network_notequal_constant():
    fake_network = lambda x: x
    fake_network.output_shape = [(1, 3)]

    expr = Constant(np.argmin)(Network("N")(Symbol("x"))) != Constant(0)
    expr.concretize(N=fake_network)

    new_expr = SubstituteCalls().visit(expr)
    expected_expr = Or(
        GreaterThan(
            Network("N")(Symbol("x"))[(0, 0)], Network("N")(Symbol("x"))[(0, 1)]
        ),
        GreaterThan(
            Network("N")(Symbol("x"))[(0, 0)], Network("N")(Symbol("x"))[(0, 2)]
        ),
    )
    assert new_expr is not expr
    assert new_expr.is_equivalent(expected_expr)


def test_argmin_concrete_network_notequal_symbol():
    fake_network = lambda x: x
    fake_network.output_shape = [(1, 3)]

    expr = Constant(np.argmin)(Network("N")(Symbol("x"))) != Symbol("y")
    expr.concretize(N=fake_network)

    new_expr = SubstituteCalls().visit(expr)
    expected_expr = Or(
        And(
            LessThanOrEqual(
                Network("N")(Symbol("x"))[(0, 0)], Network("N")(Symbol("x"))[(0, 1)]
            ),
            LessThanOrEqual(
                Network("N")(Symbol("x"))[(0, 0)], Network("N")(Symbol("x"))[(0, 2)]
            ),
            NotEqual(Symbol("y"), Constant(0)),
        ),
        And(
            LessThanOrEqual(
                Network("N")(Symbol("x"))[(0, 1)], Network("N")(Symbol("x"))[(0, 0)]
            ),
            LessThanOrEqual(
                Network("N")(Symbol("x"))[(0, 1)], Network("N")(Symbol("x"))[(0, 2)]
            ),
            NotEqual(Symbol("y"), Constant(1)),
        ),
        And(
            LessThanOrEqual(
                Network("N")(Symbol("x"))[(0, 2)], Network("N")(Symbol("x"))[(0, 0)]
            ),
            LessThanOrEqual(
                Network("N")(Symbol("x"))[(0, 2)], Network("N")(Symbol("x"))[(0, 1)]
            ),
            NotEqual(Symbol("y"), Constant(2)),
        ),
    )
    assert new_expr is not expr
    assert new_expr.is_equivalent(expected_expr)
