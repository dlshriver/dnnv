import numpy as np

from dnnv.properties.expressions import *
from dnnv.properties.transformers import SubstituteCalls


def test_abs_LessThanOrEqual_abs_cnf():
    x, y = Symbol("x"), Symbol("y")
    x.ctx.shapes[x] = ()
    x.ctx.types[x] = np.float32
    x.ctx.shapes[y] = ()
    x.ctx.types[y] = np.float32
    expr = Constant(abs)(x) <= Constant(abs)(y)
    new_expr = SubstituteCalls(form="cnf").visit(expr)
    assert new_expr is not expr
    assert new_expr.propagate_constants().is_equivalent(
        And(
            Or(
                LessThanOrEqual(x, y),
                LessThan(x, Constant(0)),
                LessThan(y, Constant(0)),
            ),
            Or(
                LessThanOrEqual(Negation(x), y),
                GreaterThanOrEqual(x, Constant(0)),
                LessThan(y, Constant(0)),
            ),
            Or(
                LessThanOrEqual(x, Negation(y)),
                LessThan(x, Constant(0)),
                GreaterThanOrEqual(y, Constant(0)),
            ),
            Or(
                LessThanOrEqual(Negation(x), Negation(y)),
                GreaterThanOrEqual(x, Constant(0)),
                GreaterThanOrEqual(y, Constant(0)),
            ),
        )
    )


def test_abs_LessThanOrEqual_abs_dnf():
    x, y = Symbol("x"), Symbol("y")
    x.ctx.shapes[x] = ()
    x.ctx.types[x] = np.float32
    x.ctx.shapes[y] = ()
    x.ctx.types[y] = np.float32
    expr = Constant(abs)(x) <= Constant(abs)(y)
    new_expr = SubstituteCalls(form="dnf").visit(expr)
    assert new_expr is not expr
    assert new_expr.propagate_constants().is_equivalent(
        Or(
            And(
                LessThanOrEqual(Negation(x), Negation(y)),
                LessThan(x, Constant(0)),
                LessThan(y, Constant(0)),
            ),
            And(
                LessThanOrEqual(x, Negation(y)),
                GreaterThanOrEqual(x, Constant(0)),
                LessThan(y, Constant(0)),
            ),
            And(
                LessThanOrEqual(Negation(x), y),
                LessThan(x, Constant(0)),
                GreaterThanOrEqual(y, Constant(0)),
            ),
            And(
                LessThanOrEqual(x, y),
                GreaterThanOrEqual(x, Constant(0)),
                GreaterThanOrEqual(y, Constant(0)),
            ),
        )
    )


def test_abs_LessThanOrEqual_abs_cnf_broadcasting():
    x, y = Symbol("x"), Symbol("y")
    x.ctx.shapes[x] = (2,)
    x.ctx.types[x] = np.float32
    x.ctx.shapes[y] = ()
    x.ctx.types[y] = np.float32
    x_0 = x[Constant((0,))]
    x_1 = x[Constant((1,))]
    y_0 = Constant(np.broadcast_to)(y, Constant((2,)))[Constant((0,))]
    y_1 = Constant(np.broadcast_to)(y, Constant((2,)))[Constant((1,))]
    expr = Constant(abs)(x) <= Constant(abs)(y)
    new_expr = SubstituteCalls(form="cnf").visit(expr)
    assert new_expr is not expr
    assert new_expr.propagate_constants().is_equivalent(
        And(
            Or(
                LessThanOrEqual(x_0, y_0),
                LessThan(x_0, Constant(0)),
                LessThan(y_0, Constant(0)),
            ),
            Or(
                LessThanOrEqual(Negation(x_0), y_0),
                GreaterThanOrEqual(x_0, Constant(0)),
                LessThan(y_0, Constant(0)),
            ),
            Or(
                LessThanOrEqual(x_0, Negation(y_0)),
                LessThan(x_0, Constant(0)),
                GreaterThanOrEqual(y_0, Constant(0)),
            ),
            Or(
                LessThanOrEqual(Negation(x_0), Negation(y_0)),
                GreaterThanOrEqual(x_0, Constant(0)),
                GreaterThanOrEqual(y_0, Constant(0)),
            ),
            Or(
                LessThanOrEqual(x_1, y_1),
                LessThan(x_1, Constant(0)),
                LessThan(y_1, Constant(0)),
            ),
            Or(
                LessThanOrEqual(Negation(x_1), y_1),
                GreaterThanOrEqual(x_1, Constant(0)),
                LessThan(y_1, Constant(0)),
            ),
            Or(
                LessThanOrEqual(x_1, Negation(y_1)),
                LessThan(x_1, Constant(0)),
                GreaterThanOrEqual(y_1, Constant(0)),
            ),
            Or(
                LessThanOrEqual(Negation(x_1), Negation(y_1)),
                GreaterThanOrEqual(x_1, Constant(0)),
                GreaterThanOrEqual(y_1, Constant(0)),
            ),
        )
    )

    x.ctx.reset()
    x, y = Symbol("x"), Symbol("y")
    x.ctx.shapes[x] = ()
    x.ctx.types[x] = np.float32
    x.ctx.shapes[y] = (2,)
    x.ctx.types[y] = np.float32
    x_0 = Constant(np.broadcast_to)(x, Constant((2,)))[Constant((0,))]
    x_1 = Constant(np.broadcast_to)(x, Constant((2,)))[Constant((1,))]
    y_0 = y[Constant((0,))]
    y_1 = y[Constant((1,))]
    expr = Constant(abs)(x) <= Constant(abs)(y)
    new_expr = SubstituteCalls(form="cnf").visit(expr)
    assert new_expr is not expr
    assert new_expr.propagate_constants().is_equivalent(
        And(
            Or(
                LessThanOrEqual(x_0, y_0),
                LessThan(x_0, Constant(0)),
                LessThan(y_0, Constant(0)),
            ),
            Or(
                LessThanOrEqual(Negation(x_0), y_0),
                GreaterThanOrEqual(x_0, Constant(0)),
                LessThan(y_0, Constant(0)),
            ),
            Or(
                LessThanOrEqual(x_0, Negation(y_0)),
                LessThan(x_0, Constant(0)),
                GreaterThanOrEqual(y_0, Constant(0)),
            ),
            Or(
                LessThanOrEqual(Negation(x_0), Negation(y_0)),
                GreaterThanOrEqual(x_0, Constant(0)),
                GreaterThanOrEqual(y_0, Constant(0)),
            ),
            Or(
                LessThanOrEqual(x_1, y_1),
                LessThan(x_1, Constant(0)),
                LessThan(y_1, Constant(0)),
            ),
            Or(
                LessThanOrEqual(Negation(x_1), y_1),
                GreaterThanOrEqual(x_1, Constant(0)),
                LessThan(y_1, Constant(0)),
            ),
            Or(
                LessThanOrEqual(x_1, Negation(y_1)),
                LessThan(x_1, Constant(0)),
                GreaterThanOrEqual(y_1, Constant(0)),
            ),
            Or(
                LessThanOrEqual(Negation(x_1), Negation(y_1)),
                GreaterThanOrEqual(x_1, Constant(0)),
                GreaterThanOrEqual(y_1, Constant(0)),
            ),
        )
    )


def test_abs_LessThanOrEqual_dnf():
    x = Symbol("x")
    x.ctx.shapes[x] = ()
    x.ctx.types[x] = np.float32
    expr = Constant(abs)(Symbol("x")) <= Constant(5)
    new_expr = SubstituteCalls(form="dnf").visit(expr)
    assert new_expr is not expr
    assert new_expr.propagate_constants().is_equivalent(
        Or(
            And(
                LessThanOrEqual(Negation(Symbol("x")), Constant(5)),
                LessThan(Symbol("x"), Constant(0)),
            ),
            And(
                LessThanOrEqual(Symbol("x"), Constant(5)),
                GreaterThanOrEqual(Symbol("x"), Constant(0)),
            ),
        )
    )

    expr = Constant(15) <= Constant(abs)(Symbol("x"))
    new_expr = SubstituteCalls(form="dnf").visit(expr)
    assert new_expr is not expr
    assert new_expr.propagate_constants().is_equivalent(
        Or(
            And(
                LessThanOrEqual(Constant(15), Negation(Symbol("x"))),
                LessThan(Symbol("x"), Constant(0)),
            ),
            And(
                LessThanOrEqual(Constant(15), Symbol("x")),
                GreaterThanOrEqual(Symbol("x"), Constant(0)),
            ),
        ),
    )


def test_abs_LessThanOrEqual_cnf():
    x = Symbol("x")
    x.ctx.shapes[x] = ()
    x.ctx.types[x] = np.float32
    expr = Constant(abs)(Symbol("x")) <= Constant(5)
    new_expr = SubstituteCalls(form="cnf").visit(expr)
    assert new_expr is not expr
    assert new_expr.propagate_constants().is_equivalent(
        And(
            Or(
                LessThanOrEqual(Symbol("x"), Constant(5)),
                LessThan(Symbol("x"), Constant(0)),
            ),
            Or(
                LessThanOrEqual(Negation(Symbol("x")), Constant(5)),
                GreaterThanOrEqual(Symbol("x"), Constant(0)),
            ),
        )
    )

    expr = Constant(15) <= Constant(abs)(Symbol("x"))
    new_expr = SubstituteCalls(form="cnf").visit(expr)
    assert new_expr is not expr
    assert new_expr.propagate_constants().is_equivalent(
        And(
            Or(
                LessThanOrEqual(Constant(15), Symbol("x")),
                LessThan(Symbol("x"), Constant(0)),
            ),
            Or(
                LessThanOrEqual(Constant(15), Negation(Symbol("x"))),
                GreaterThanOrEqual(Symbol("x"), Constant(0)),
            ),
        ),
    )


def test_abs_LessThanOrEqual_cnf_broadcasting_call():
    x = Symbol("x")
    x.ctx.shapes[x] = ()
    x.ctx.types[x] = np.float32
    expr = Constant(abs)(x) <= Constant(np.array([1, 2, 3]))
    new_expr = SubstituteCalls(form="cnf").visit(expr)
    assert new_expr is not expr
    assert new_expr.propagate_constants().is_equivalent(
        And(
            Or(
                LessThanOrEqual(
                    Negation(
                        Constant(np.broadcast_to)(x, Constant((3,)))[Constant((0,))]
                    ),
                    Constant(1),
                ),
                GreaterThanOrEqual(
                    Constant(np.broadcast_to)(x, Constant((3,)))[Constant((0,))],
                    Constant(0),
                ),
            ),
            Or(
                LessThanOrEqual(
                    Negation(
                        Constant(np.broadcast_to)(x, Constant((3,)))[Constant((1,))]
                    ),
                    Constant(2),
                ),
                GreaterThanOrEqual(
                    Constant(np.broadcast_to)(x, Constant((3,)))[Constant((1,))],
                    Constant(0),
                ),
            ),
            Or(
                LessThanOrEqual(
                    Negation(
                        Constant(np.broadcast_to)(x, Constant((3,)))[Constant((2,))]
                    ),
                    Constant(3),
                ),
                GreaterThanOrEqual(
                    Constant(np.broadcast_to)(x, Constant((3,)))[Constant((2,))],
                    Constant(0),
                ),
            ),
            Or(
                LessThanOrEqual(
                    Constant(np.broadcast_to)(x, Constant((3,)))[Constant((0,))],
                    Constant(1),
                ),
                LessThan(
                    Constant(np.broadcast_to)(x, Constant((3,)))[Constant((0,))],
                    Constant(0),
                ),
            ),
            Or(
                LessThanOrEqual(
                    Constant(np.broadcast_to)(x, Constant((3,)))[Constant((1,))],
                    Constant(2),
                ),
                LessThan(
                    Constant(np.broadcast_to)(x, Constant((3,)))[Constant((1,))],
                    Constant(0),
                ),
            ),
            Or(
                LessThanOrEqual(
                    Constant(np.broadcast_to)(x, Constant((3,)))[Constant((2,))],
                    Constant(3),
                ),
                LessThan(
                    Constant(np.broadcast_to)(x, Constant((3,)))[Constant((2,))],
                    Constant(0),
                ),
            ),
        )
    )

    expr = Constant(np.array([1, 2, 3])) <= Constant(abs)(x)
    new_expr = SubstituteCalls(form="cnf").visit(expr)
    assert new_expr is not expr
    assert new_expr.propagate_constants().is_equivalent(
        And(
            Or(
                LessThanOrEqual(
                    Constant(1),
                    Negation(
                        Constant(np.broadcast_to)(x, Constant((3,)))[Constant((0,))]
                    ),
                ),
                GreaterThanOrEqual(
                    Constant(np.broadcast_to)(x, Constant((3,)))[Constant((0,))],
                    Constant(0),
                ),
            ),
            Or(
                LessThanOrEqual(
                    Constant(2),
                    Negation(
                        Constant(np.broadcast_to)(x, Constant((3,)))[Constant((1,))]
                    ),
                ),
                GreaterThanOrEqual(
                    Constant(np.broadcast_to)(x, Constant((3,)))[Constant((1,))],
                    Constant(0),
                ),
            ),
            Or(
                LessThanOrEqual(
                    Constant(3),
                    Negation(
                        Constant(np.broadcast_to)(x, Constant((3,)))[Constant((2,))]
                    ),
                ),
                GreaterThanOrEqual(
                    Constant(np.broadcast_to)(x, Constant((3,)))[Constant((2,))],
                    Constant(0),
                ),
            ),
            Or(
                LessThanOrEqual(
                    Constant(1),
                    Constant(np.broadcast_to)(x, Constant((3,)))[Constant((0,))],
                ),
                LessThan(
                    Constant(np.broadcast_to)(x, Constant((3,)))[Constant((0,))],
                    Constant(0),
                ),
            ),
            Or(
                LessThanOrEqual(
                    Constant(2),
                    Constant(np.broadcast_to)(x, Constant((3,)))[Constant((1,))],
                ),
                LessThan(
                    Constant(np.broadcast_to)(x, Constant((3,)))[Constant((1,))],
                    Constant(0),
                ),
            ),
            Or(
                LessThanOrEqual(
                    Constant(3),
                    Constant(np.broadcast_to)(x, Constant((3,)))[Constant((2,))],
                ),
                LessThan(
                    Constant(np.broadcast_to)(x, Constant((3,)))[Constant((2,))],
                    Constant(0),
                ),
            ),
        )
    )


def test_abs_LessThanOrEqual_cnf_broadcasting_noncall():
    x = Symbol("x")
    x.ctx.shapes[x] = (3,)
    x.ctx.types[x] = np.float32
    expr = Constant(abs)(x) <= Constant(1)
    new_expr = SubstituteCalls(form="cnf").visit(expr)
    assert new_expr is not expr
    assert new_expr.propagate_constants().is_equivalent(
        And(
            Or(
                LessThanOrEqual(Negation(x[Constant((0,))]), Constant(1)),
                GreaterThanOrEqual(x[Constant((0,))], Constant(0)),
            ),
            Or(
                LessThanOrEqual(Negation(x[Constant((1,))]), Constant(1)),
                GreaterThanOrEqual(x[Constant((1,))], Constant(0)),
            ),
            Or(
                LessThanOrEqual(Negation(x[Constant((2,))]), Constant(1)),
                GreaterThanOrEqual(x[Constant((2,))], Constant(0)),
            ),
            Or(
                LessThanOrEqual(x[Constant((0,))], Constant(1)),
                LessThan(x[Constant((0,))], Constant(0)),
            ),
            Or(
                LessThanOrEqual(x[Constant((1,))], Constant(1)),
                LessThan(x[Constant((1,))], Constant(0)),
            ),
            Or(
                LessThanOrEqual(x[Constant((2,))], Constant(1)),
                LessThan(x[Constant((2,))], Constant(0)),
            ),
        )
    )

    expr = Constant(1) <= Constant(abs)(x)
    new_expr = SubstituteCalls(form="cnf").visit(expr)
    assert new_expr is not expr
    assert new_expr.propagate_constants().is_equivalent(
        And(
            Or(
                LessThanOrEqual(Constant(1), Negation(x[Constant((0,))])),
                GreaterThanOrEqual(x[Constant((0,))], Constant(0)),
            ),
            Or(
                LessThanOrEqual(Constant(1), Negation(x[Constant((1,))])),
                GreaterThanOrEqual(x[Constant((1,))], Constant(0)),
            ),
            Or(
                LessThanOrEqual(Constant(1), Negation(x[Constant((2,))])),
                GreaterThanOrEqual(x[Constant((2,))], Constant(0)),
            ),
            Or(
                LessThanOrEqual(Constant(1), x[Constant((0,))]),
                LessThan(x[Constant((0,))], Constant(0)),
            ),
            Or(
                LessThanOrEqual(Constant(1), x[Constant((1,))]),
                LessThan(x[Constant((1,))], Constant(0)),
            ),
            Or(
                LessThanOrEqual(Constant(1), x[Constant((2,))]),
                LessThan(x[Constant((2,))], Constant(0)),
            ),
        )
    )


def test_abs_LessThanOrEqual_negative():
    x = Symbol("x")
    x.ctx.shapes[x] = ()
    x.ctx.types[x] = np.float32
    expr = Constant(abs)(Symbol("x")) <= Constant(-13)
    new_expr = SubstituteCalls(form="dnf").visit(expr)
    assert new_expr is not expr
    assert new_expr.propagate_constants() is Constant(False)

    expr = Constant(-13) <= Constant(abs)(Symbol("x"))
    new_expr = SubstituteCalls(form="dnf").visit(expr)
    assert new_expr is not expr
    assert new_expr.propagate_constants() is Constant(False)
