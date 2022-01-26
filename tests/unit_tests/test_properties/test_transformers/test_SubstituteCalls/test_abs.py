import numpy as np

from dnnv.properties.expressions import *
from dnnv.properties.transformers import SubstituteCalls


def test_abs():
    expr = Constant(abs)(Symbol("x"))
    new_expr = SubstituteCalls().visit(expr)
    assert new_expr is not expr
    assert new_expr.is_equivalent(
        IfThenElse(Symbol("x") >= Constant(0), Symbol("x"), -Symbol("x"))
    )


def test_numpy_abs():
    expr = Constant(np.abs)(Symbol("x"))
    new_expr = SubstituteCalls().visit(expr)
    assert new_expr is not expr
    assert new_expr.is_equivalent(
        IfThenElse(Symbol("x") >= Constant(0), Symbol("x"), -Symbol("x"))
    )


def test_abs_constant():
    expr = Call(Constant(abs), (Constant(10),), {})
    new_expr = SubstituteCalls().visit(expr)
    assert new_expr is not expr
    assert new_expr is Constant(10)


def test_abs_Equal():
    expr = Constant(abs)(Symbol("x")) == Constant(5)
    new_expr = SubstituteCalls().visit(expr)
    assert new_expr is not expr
    assert new_expr.is_equivalent(
        And(
            GreaterThanOrEqual(Constant(5), Constant(0)),
            Or(
                And(
                    Equal(Negation(Symbol("x")), Constant(5)),
                    LessThan(Symbol("x"), Constant(0)),
                ),
                And(
                    Equal(Symbol("x"), Constant(5)),
                    GreaterThanOrEqual(Symbol("x"), Constant(0)),
                ),
            ),
        )
    )

    expr = Constant(15) == Constant(abs)(Symbol("x"))
    new_expr = SubstituteCalls().visit(expr)
    assert new_expr is not expr
    assert new_expr.is_equivalent(
        And(
            GreaterThanOrEqual(Constant(15), Constant(0)),
            Or(
                And(
                    Equal(Constant(15), Negation(Symbol("x"))),
                    LessThan(Symbol("x"), Constant(0)),
                ),
                And(
                    Equal(Constant(15), Symbol("x")),
                    GreaterThanOrEqual(Symbol("x"), Constant(0)),
                ),
            ),
        )
    )


def test_abs_Equal_negative():
    expr = Constant(abs)(Symbol("x")) == Constant(-13)
    new_expr = SubstituteCalls().visit(expr)
    assert new_expr is not expr
    assert new_expr.propagate_constants() is Constant(False)


def test_abs_NotEqual():
    expr = Constant(abs)(Symbol("x")) != Constant(7)
    new_expr = SubstituteCalls().visit(expr)
    assert new_expr is not expr
    assert new_expr.is_equivalent(
        Or(
            LessThan(Constant(7), Constant(0)),
            And(
                NotEqual(Negation(Symbol("x")), Constant(7)),
                LessThan(Symbol("x"), Constant(0)),
            ),
            And(
                NotEqual(Symbol("x"), Constant(7)),
                GreaterThanOrEqual(Symbol("x"), Constant(0)),
            ),
        )
    )

    expr = Constant(8) != Constant(abs)(Symbol("x"))
    new_expr = SubstituteCalls().visit(expr)
    assert new_expr is not expr
    assert new_expr.is_equivalent(
        Or(
            LessThan(Constant(8), Constant(0)),
            And(
                NotEqual(Constant(8), Negation(Symbol("x"))),
                LessThan(Symbol("x"), Constant(0)),
            ),
            And(
                NotEqual(Constant(8), Symbol("x")),
                GreaterThanOrEqual(Symbol("x"), Constant(0)),
            ),
        )
    )


def test_abs_NotEqual_negative():
    expr = Constant(abs)(Symbol("x")) != Constant(-13)
    new_expr = SubstituteCalls().visit(expr)
    assert new_expr is not expr
    assert new_expr.propagate_constants() is Constant(True)


def test_abs_GreaterThan():
    expr = Constant(abs)(Symbol("x")) > Constant(1)
    new_expr = SubstituteCalls().visit(expr)
    assert new_expr is not expr
    assert new_expr.is_equivalent(
        Or(
            LessThan(Constant(1), Constant(0)),
            And(
                GreaterThan(Negation(Symbol("x")), Constant(1)),
                LessThan(Symbol("x"), Constant(0)),
            ),
            And(
                GreaterThan(Symbol("x"), Constant(1)),
                GreaterThanOrEqual(Symbol("x"), Constant(0)),
            ),
        )
    )

    expr = Constant(12) > Constant(abs)(Symbol("x"))
    new_expr = SubstituteCalls().visit(expr)
    assert new_expr is not expr
    assert new_expr.is_equivalent(
        Or(
            LessThan(Constant(12), Constant(0)),
            And(
                GreaterThan(Constant(12), Negation(Symbol("x"))),
                LessThan(Symbol("x"), Constant(0)),
            ),
            And(
                GreaterThan(Constant(12), Symbol("x")),
                GreaterThanOrEqual(Symbol("x"), Constant(0)),
            ),
        )
    )


def test_abs_GreaterThan_negative():
    expr = Constant(abs)(Symbol("x")) > Constant(-13)
    new_expr = SubstituteCalls().visit(expr)
    assert new_expr is not expr
    assert new_expr.propagate_constants() is Constant(True)


def test_abs_GreaterThanOrEqual():
    expr = Constant(abs)(Symbol("x")) >= Constant(1)
    new_expr = SubstituteCalls().visit(expr)
    assert new_expr is not expr
    assert new_expr.is_equivalent(
        Or(
            LessThan(Constant(1), Constant(0)),
            And(
                GreaterThanOrEqual(Negation(Symbol("x")), Constant(1)),
                LessThan(Symbol("x"), Constant(0)),
            ),
            And(
                GreaterThanOrEqual(Symbol("x"), Constant(1)),
                GreaterThanOrEqual(Symbol("x"), Constant(0)),
            ),
        )
    )

    expr = Constant(12) >= Constant(abs)(Symbol("x"))
    new_expr = SubstituteCalls().visit(expr)
    assert new_expr is not expr
    assert new_expr.is_equivalent(
        Or(
            LessThan(Constant(12), Constant(0)),
            And(
                GreaterThanOrEqual(Constant(12), Negation(Symbol("x"))),
                LessThan(Symbol("x"), Constant(0)),
            ),
            And(
                GreaterThanOrEqual(Constant(12), Symbol("x")),
                GreaterThanOrEqual(Symbol("x"), Constant(0)),
            ),
        )
    )


def test_abs_GreaterThanOrEqual_negative():
    expr = Constant(abs)(Symbol("x")) >= Constant(-13)
    new_expr = SubstituteCalls().visit(expr)
    assert new_expr is not expr
    assert new_expr.propagate_constants() is Constant(True)


def test_abs_LessThan():
    expr = Constant(abs)(Symbol("x")) < Constant(1)
    new_expr = SubstituteCalls().visit(expr)
    assert new_expr is not expr
    assert new_expr.is_equivalent(
        And(
            GreaterThanOrEqual(Constant(1), Constant(0)),
            Or(
                And(
                    LessThan(Negation(Symbol("x")), Constant(1)),
                    LessThan(Symbol("x"), Constant(0)),
                ),
                And(
                    LessThan(Symbol("x"), Constant(1)),
                    GreaterThanOrEqual(Symbol("x"), Constant(0)),
                ),
            ),
        )
    )

    expr = Constant(12) < Constant(abs)(Symbol("x"))
    new_expr = SubstituteCalls().visit(expr)
    assert new_expr is not expr
    assert new_expr.is_equivalent(
        And(
            GreaterThanOrEqual(Constant(12), Constant(0)),
            Or(
                And(
                    LessThan(Constant(12), Negation(Symbol("x"))),
                    LessThan(Symbol("x"), Constant(0)),
                ),
                And(
                    LessThan(Constant(12), Symbol("x")),
                    GreaterThanOrEqual(Symbol("x"), Constant(0)),
                ),
            ),
        )
    )


def test_abs_LessThan_negative():
    expr = Constant(abs)(Symbol("x")) < Constant(-13)
    new_expr = SubstituteCalls().visit(expr)
    assert new_expr is not expr
    assert new_expr.propagate_constants() is Constant(False)


def test_abs_LessThanOrEqual():
    expr = Constant(abs)(Symbol("x")) <= Constant(1)
    new_expr = SubstituteCalls().visit(expr)
    assert new_expr is not expr
    assert new_expr.is_equivalent(
        And(
            GreaterThanOrEqual(Constant(1), Constant(0)),
            Or(
                And(
                    LessThanOrEqual(Negation(Symbol("x")), Constant(1)),
                    LessThan(Symbol("x"), Constant(0)),
                ),
                And(
                    LessThanOrEqual(Symbol("x"), Constant(1)),
                    GreaterThanOrEqual(Symbol("x"), Constant(0)),
                ),
            ),
        )
    )

    expr = Constant(12) <= Constant(abs)(Symbol("x"))
    new_expr = SubstituteCalls().visit(expr)
    assert new_expr is not expr
    assert new_expr.is_equivalent(
        And(
            GreaterThanOrEqual(Constant(12), Constant(0)),
            Or(
                And(
                    LessThanOrEqual(Constant(12), Negation(Symbol("x"))),
                    LessThan(Symbol("x"), Constant(0)),
                ),
                And(
                    LessThanOrEqual(Constant(12), Symbol("x")),
                    GreaterThanOrEqual(Symbol("x"), Constant(0)),
                ),
            ),
        )
    )


def test_abs_LessThanOrEqual_negative():
    expr = Constant(abs)(Symbol("x")) <= Constant(-13)
    new_expr = SubstituteCalls().visit(expr)
    assert new_expr is not expr
    assert new_expr.propagate_constants() is Constant(False)
