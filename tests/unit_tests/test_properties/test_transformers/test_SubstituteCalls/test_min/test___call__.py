import re

import numpy as np
import pytest

from dnnv.errors import DNNVError
from dnnv.properties.expressions import *
from dnnv.properties.transformers import SubstituteCalls


def test_numpy_min():
    x = Symbol("x")
    x.ctx.shapes[x] = (2, 3)
    x.ctx.types[x] = np.float32
    expr = Constant(np.min)(x)
    new_expr = SubstituteCalls().visit(expr)
    assert new_expr is not expr
    assert new_expr.is_equivalent(
        IfThenElse(
            And(
                LessThanOrEqual(Symbol("x")[(0, 0)], Symbol("x")[(0, 1)]),
                LessThanOrEqual(Symbol("x")[(0, 0)], Symbol("x")[(0, 2)]),
                LessThanOrEqual(Symbol("x")[(0, 0)], Symbol("x")[(1, 0)]),
                LessThanOrEqual(Symbol("x")[(0, 0)], Symbol("x")[(1, 1)]),
                LessThanOrEqual(Symbol("x")[(0, 0)], Symbol("x")[(1, 2)]),
            ),
            Symbol("x")[(0, 0)],
            IfThenElse(
                And(
                    LessThanOrEqual(Symbol("x")[(0, 1)], Symbol("x")[(0, 2)]),
                    LessThanOrEqual(Symbol("x")[(0, 1)], Symbol("x")[(1, 0)]),
                    LessThanOrEqual(Symbol("x")[(0, 1)], Symbol("x")[(1, 1)]),
                    LessThanOrEqual(Symbol("x")[(0, 1)], Symbol("x")[(1, 2)]),
                ),
                Symbol("x")[(0, 1)],
                IfThenElse(
                    And(
                        LessThanOrEqual(Symbol("x")[(0, 2)], Symbol("x")[(1, 0)]),
                        LessThanOrEqual(Symbol("x")[(0, 2)], Symbol("x")[(1, 1)]),
                        LessThanOrEqual(Symbol("x")[(0, 2)], Symbol("x")[(1, 2)]),
                    ),
                    Symbol("x")[(0, 2)],
                    IfThenElse(
                        And(
                            LessThanOrEqual(Symbol("x")[(1, 0)], Symbol("x")[(1, 1)]),
                            LessThanOrEqual(Symbol("x")[(1, 0)], Symbol("x")[(1, 2)]),
                        ),
                        Symbol("x")[(1, 0)],
                        IfThenElse(
                            And(
                                LessThanOrEqual(
                                    Symbol("x")[(1, 1)], Symbol("x")[(1, 2)]
                                ),
                            ),
                            Symbol("x")[(1, 1)],
                            Symbol("x")[(1, 2)],
                        ),
                    ),
                ),
            ),
        )
    )


def test_numpy_min_initial():
    x = Symbol("x")
    x.ctx.shapes[x] = (2, 3)
    x.ctx.types[x] = np.float32
    expr = Constant(np.min)(x, initial=Constant(-1.0))
    new_expr = SubstituteCalls().visit(expr)
    assert new_expr is not expr
    assert new_expr.is_equivalent(
        IfThenElse(
            And(
                LessThanOrEqual(Symbol("x")[(0, 0)], Constant(-1.0)),
                LessThanOrEqual(Symbol("x")[(0, 0)], Symbol("x")[(0, 1)]),
                LessThanOrEqual(Symbol("x")[(0, 0)], Symbol("x")[(0, 2)]),
                LessThanOrEqual(Symbol("x")[(0, 0)], Symbol("x")[(1, 0)]),
                LessThanOrEqual(Symbol("x")[(0, 0)], Symbol("x")[(1, 1)]),
                LessThanOrEqual(Symbol("x")[(0, 0)], Symbol("x")[(1, 2)]),
            ),
            Symbol("x")[(0, 0)],
            IfThenElse(
                And(
                    LessThanOrEqual(Symbol("x")[(0, 1)], Constant(-1.0)),
                    LessThanOrEqual(Symbol("x")[(0, 1)], Symbol("x")[(0, 2)]),
                    LessThanOrEqual(Symbol("x")[(0, 1)], Symbol("x")[(1, 0)]),
                    LessThanOrEqual(Symbol("x")[(0, 1)], Symbol("x")[(1, 1)]),
                    LessThanOrEqual(Symbol("x")[(0, 1)], Symbol("x")[(1, 2)]),
                ),
                Symbol("x")[(0, 1)],
                IfThenElse(
                    And(
                        LessThanOrEqual(Symbol("x")[(0, 2)], Constant(-1.0)),
                        LessThanOrEqual(Symbol("x")[(0, 2)], Symbol("x")[(1, 0)]),
                        LessThanOrEqual(Symbol("x")[(0, 2)], Symbol("x")[(1, 1)]),
                        LessThanOrEqual(Symbol("x")[(0, 2)], Symbol("x")[(1, 2)]),
                    ),
                    Symbol("x")[(0, 2)],
                    IfThenElse(
                        And(
                            LessThanOrEqual(Symbol("x")[(1, 0)], Constant(-1.0)),
                            LessThanOrEqual(Symbol("x")[(1, 0)], Symbol("x")[(1, 1)]),
                            LessThanOrEqual(Symbol("x")[(1, 0)], Symbol("x")[(1, 2)]),
                        ),
                        Symbol("x")[(1, 0)],
                        IfThenElse(
                            And(
                                LessThanOrEqual(Symbol("x")[(1, 1)], Constant(-1.0)),
                                LessThanOrEqual(
                                    Symbol("x")[(1, 1)], Symbol("x")[(1, 2)]
                                ),
                            ),
                            Symbol("x")[(1, 1)],
                            IfThenElse(
                                And(
                                    LessThanOrEqual(Symbol("x")[(1, 2)], Constant(-1.0))
                                ),
                                Symbol("x")[(1, 2)],
                                Constant(-1.0),
                            ),
                        ),
                    ),
                ),
            ),
        )
    )


def test_numpy_min_empty_with_initial():
    x = Symbol("x")
    x.ctx.shapes[x] = (0,)
    x.ctx.types[x] = np.float32
    expr = Constant(np.min)(x, initial=Constant(-1.0))
    new_expr = SubstituteCalls().visit(expr)
    assert new_expr is Constant(-1.0)


def test_numpy_min_constant():
    a = np.random.randn(3, 4, 5)
    expr = Constant(np.min)(Constant(a))
    new_expr = SubstituteCalls().visit(expr)
    assert new_expr is not expr
    assert new_expr is Constant(np.min(a))


def test_numpy_min_constant_with_initial():
    a = np.random.rand(3, 4, 5)
    expr = Constant(np.min)(Constant(a), initial=Constant(-1.0))
    new_expr = SubstituteCalls().visit(expr)
    assert new_expr is not expr
    assert new_expr is Constant(np.float64(-1.0))

    expr = Constant(np.min)(Constant(a), initial=Constant(0.1))
    new_expr = SubstituteCalls().visit(expr)
    assert new_expr is not expr
    assert new_expr is Constant(np.min(a, initial=0.1))


def test_numpy_min_noshape():
    x = Symbol("x")
    expr = Constant(np.min)(x)
    new_expr = SubstituteCalls().visit(expr)
    assert new_expr.is_equivalent(expr)


def test_builtin_min():
    x = Symbol("x")
    x.ctx.shapes[x] = (4,)
    x.ctx.types[x] = np.float32
    expr = Constant(min)(x)
    new_expr = SubstituteCalls().visit(expr)
    assert new_expr is not expr
    assert new_expr.is_equivalent(
        IfThenElse(
            And(
                LessThanOrEqual(Symbol("x")[(0,)], Symbol("x")[(1,)]),
                LessThanOrEqual(Symbol("x")[(0,)], Symbol("x")[(2,)]),
                LessThanOrEqual(Symbol("x")[(0,)], Symbol("x")[(3,)]),
            ),
            Symbol("x")[(0,)],
            IfThenElse(
                And(
                    LessThanOrEqual(Symbol("x")[(1,)], Symbol("x")[(2,)]),
                    LessThanOrEqual(Symbol("x")[(1,)], Symbol("x")[(3,)]),
                ),
                Symbol("x")[(1,)],
                IfThenElse(
                    And(
                        LessThanOrEqual(Symbol("x")[(2,)], Symbol("x")[(3,)]),
                    ),
                    Symbol("x")[(2,)],
                    Symbol("x")[(3,)],
                ),
            ),
        )
    )


def test_builtin_min_empty_no_default():
    x = Symbol("x")
    x.ctx.shapes[x] = (0,)
    x.ctx.types[x] = np.float32
    expr = Constant(min)(x)
    with pytest.raises(ValueError, match=re.escape("min() arg is an empty sequence")):
        _ = SubstituteCalls().visit(expr)


def test_builtin_min_empty_with_default():
    x = Symbol("x")
    x.ctx.shapes[x] = (0,)
    x.ctx.types[x] = np.float32
    expr = Constant(min)(x, default=Constant(-1.0))
    new_expr = SubstituteCalls().visit(expr)
    assert new_expr is Constant(-1.0)


def test_builtin_min_constant():
    a = np.random.randn(5)
    expr = Constant(min)(Constant(a))
    new_expr = SubstituteCalls().visit(expr)
    assert new_expr is not expr
    assert new_expr is Constant(min(a))


def test_builtin_min_constant_with_default():
    a = np.random.randn(5)
    expr = Constant(min)(Constant(a), default=Constant(-1.0))
    new_expr = SubstituteCalls().visit(expr)
    assert new_expr is not expr
    assert new_expr is Constant(min(a, default=-1))

    expr = Constant(min)(Constant(a), default=Constant(0.1))
    new_expr = SubstituteCalls().visit(expr)
    assert new_expr is not expr
    assert new_expr is Constant(min(a, default=0.1))


def test_builtin_min_noshape():
    x = Symbol("x")
    expr = Constant(min)(x)
    new_expr = SubstituteCalls().visit(expr)
    assert new_expr.is_equivalent(expr)


def test_builtin_muliarg_min():
    a = Symbol("a")
    a.ctx.shapes[a] = ()
    a.ctx.types[a] = np.float32
    b = Symbol("b")
    b.ctx.shapes[b] = ()
    b.ctx.types[b] = np.float32
    c = Symbol("c")
    c.ctx.shapes[c] = ()
    c.ctx.types[c] = np.float32
    expr = Constant(min)(a, b, c)
    new_expr = SubstituteCalls().visit(expr)
    assert new_expr is not expr
    assert new_expr.is_equivalent(
        IfThenElse(
            And(LessThanOrEqual(a, b), LessThanOrEqual(a, c)),
            a,
            IfThenElse(And(LessThanOrEqual(b, c)), b, c),
        )
    )


def test_builtin_multiarg_min_constant():
    a = np.random.randn()
    b = np.random.randn()
    c = np.random.randn()
    expr = Constant(min)(Constant(a), Constant(b), Constant(c))
    new_expr = SubstituteCalls().visit(expr)
    assert new_expr is not expr
    assert new_expr is Constant(min(a, b, c))
