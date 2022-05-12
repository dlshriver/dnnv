import re

import numpy as np
import pytest

from dnnv.properties.expressions import *
from dnnv.properties.transformers import SubstituteCalls


def test_numpy_max():
    x = Symbol("x")
    x.ctx.shapes[x] = (2, 3)
    x.ctx.types[x] = np.float32
    expr = Constant(np.max)(x)
    new_expr = SubstituteCalls().visit(expr)
    assert new_expr is not expr
    assert new_expr.is_equivalent(
        IfThenElse(
            And(
                GreaterThanOrEqual(Symbol("x")[(0, 0)], Symbol("x")[(0, 1)]),
                GreaterThanOrEqual(Symbol("x")[(0, 0)], Symbol("x")[(0, 2)]),
                GreaterThanOrEqual(Symbol("x")[(0, 0)], Symbol("x")[(1, 0)]),
                GreaterThanOrEqual(Symbol("x")[(0, 0)], Symbol("x")[(1, 1)]),
                GreaterThanOrEqual(Symbol("x")[(0, 0)], Symbol("x")[(1, 2)]),
            ),
            Symbol("x")[(0, 0)],
            IfThenElse(
                And(
                    GreaterThanOrEqual(Symbol("x")[(0, 1)], Symbol("x")[(0, 2)]),
                    GreaterThanOrEqual(Symbol("x")[(0, 1)], Symbol("x")[(1, 0)]),
                    GreaterThanOrEqual(Symbol("x")[(0, 1)], Symbol("x")[(1, 1)]),
                    GreaterThanOrEqual(Symbol("x")[(0, 1)], Symbol("x")[(1, 2)]),
                ),
                Symbol("x")[(0, 1)],
                IfThenElse(
                    And(
                        GreaterThanOrEqual(Symbol("x")[(0, 2)], Symbol("x")[(1, 0)]),
                        GreaterThanOrEqual(Symbol("x")[(0, 2)], Symbol("x")[(1, 1)]),
                        GreaterThanOrEqual(Symbol("x")[(0, 2)], Symbol("x")[(1, 2)]),
                    ),
                    Symbol("x")[(0, 2)],
                    IfThenElse(
                        And(
                            GreaterThanOrEqual(
                                Symbol("x")[(1, 0)], Symbol("x")[(1, 1)]
                            ),
                            GreaterThanOrEqual(
                                Symbol("x")[(1, 0)], Symbol("x")[(1, 2)]
                            ),
                        ),
                        Symbol("x")[(1, 0)],
                        IfThenElse(
                            And(
                                GreaterThanOrEqual(
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


def test_numpy_max_initial():
    x = Symbol("x")
    x.ctx.shapes[x] = (2, 3)
    x.ctx.types[x] = np.float32
    expr = Constant(np.max)(x, initial=Constant(-1.0))
    new_expr = SubstituteCalls().visit(expr)
    assert new_expr is not expr
    assert new_expr.is_equivalent(
        IfThenElse(
            And(
                GreaterThanOrEqual(Symbol("x")[(0, 0)], Constant(-1.0)),
                GreaterThanOrEqual(Symbol("x")[(0, 0)], Symbol("x")[(0, 1)]),
                GreaterThanOrEqual(Symbol("x")[(0, 0)], Symbol("x")[(0, 2)]),
                GreaterThanOrEqual(Symbol("x")[(0, 0)], Symbol("x")[(1, 0)]),
                GreaterThanOrEqual(Symbol("x")[(0, 0)], Symbol("x")[(1, 1)]),
                GreaterThanOrEqual(Symbol("x")[(0, 0)], Symbol("x")[(1, 2)]),
            ),
            Symbol("x")[(0, 0)],
            IfThenElse(
                And(
                    GreaterThanOrEqual(Symbol("x")[(0, 1)], Constant(-1.0)),
                    GreaterThanOrEqual(Symbol("x")[(0, 1)], Symbol("x")[(0, 2)]),
                    GreaterThanOrEqual(Symbol("x")[(0, 1)], Symbol("x")[(1, 0)]),
                    GreaterThanOrEqual(Symbol("x")[(0, 1)], Symbol("x")[(1, 1)]),
                    GreaterThanOrEqual(Symbol("x")[(0, 1)], Symbol("x")[(1, 2)]),
                ),
                Symbol("x")[(0, 1)],
                IfThenElse(
                    And(
                        GreaterThanOrEqual(Symbol("x")[(0, 2)], Constant(-1.0)),
                        GreaterThanOrEqual(Symbol("x")[(0, 2)], Symbol("x")[(1, 0)]),
                        GreaterThanOrEqual(Symbol("x")[(0, 2)], Symbol("x")[(1, 1)]),
                        GreaterThanOrEqual(Symbol("x")[(0, 2)], Symbol("x")[(1, 2)]),
                    ),
                    Symbol("x")[(0, 2)],
                    IfThenElse(
                        And(
                            GreaterThanOrEqual(Symbol("x")[(1, 0)], Constant(-1.0)),
                            GreaterThanOrEqual(
                                Symbol("x")[(1, 0)], Symbol("x")[(1, 1)]
                            ),
                            GreaterThanOrEqual(
                                Symbol("x")[(1, 0)], Symbol("x")[(1, 2)]
                            ),
                        ),
                        Symbol("x")[(1, 0)],
                        IfThenElse(
                            And(
                                GreaterThanOrEqual(Symbol("x")[(1, 1)], Constant(-1.0)),
                                GreaterThanOrEqual(
                                    Symbol("x")[(1, 1)], Symbol("x")[(1, 2)]
                                ),
                            ),
                            Symbol("x")[(1, 1)],
                            IfThenElse(
                                And(
                                    GreaterThanOrEqual(
                                        Symbol("x")[(1, 2)], Constant(-1.0)
                                    )
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


def test_numpy_max_empty_with_initial():
    x = Symbol("x")
    x.ctx.shapes[x] = (0,)
    x.ctx.types[x] = np.float32
    expr = Constant(np.max)(x, initial=Constant(2.0))
    new_expr = SubstituteCalls().visit(expr)
    assert new_expr is Constant(2.0)


def test_numpy_max_constant():
    a = np.random.randn(3, 4, 5)
    expr = Constant(np.max)(Constant(a))
    new_expr = SubstituteCalls().visit(expr)
    assert new_expr is not expr
    assert new_expr is Constant(np.max(a))


def test_numpy_max_constant_with_initial():
    a = np.random.rand(3, 4, 5)
    expr = Constant(np.max)(Constant(a), initial=Constant(2.0))
    new_expr = SubstituteCalls().visit(expr)
    assert new_expr is not expr
    assert new_expr is Constant(np.float64(2.0))

    expr = Constant(np.max)(Constant(a), initial=Constant(0.9))
    new_expr = SubstituteCalls().visit(expr)
    assert new_expr is not expr
    assert new_expr is Constant(np.max(a, initial=0.9))


def test_numpy_max_noshape():
    x = Symbol("x")
    expr = Constant(np.max)(x)
    new_expr = SubstituteCalls().visit(expr)
    assert new_expr.is_equivalent(expr)


def test_builtin_max():
    x = Symbol("x")
    x.ctx.shapes[x] = (4,)
    x.ctx.types[x] = np.float32
    expr = Constant(max)(x)
    new_expr = SubstituteCalls().visit(expr)
    assert new_expr is not expr
    assert new_expr.is_equivalent(
        IfThenElse(
            And(
                GreaterThanOrEqual(Symbol("x")[(0,)], Symbol("x")[(1,)]),
                GreaterThanOrEqual(Symbol("x")[(0,)], Symbol("x")[(2,)]),
                GreaterThanOrEqual(Symbol("x")[(0,)], Symbol("x")[(3,)]),
            ),
            Symbol("x")[(0,)],
            IfThenElse(
                And(
                    GreaterThanOrEqual(Symbol("x")[(1,)], Symbol("x")[(2,)]),
                    GreaterThanOrEqual(Symbol("x")[(1,)], Symbol("x")[(3,)]),
                ),
                Symbol("x")[(1,)],
                IfThenElse(
                    And(
                        GreaterThanOrEqual(Symbol("x")[(2,)], Symbol("x")[(3,)]),
                    ),
                    Symbol("x")[(2,)],
                    Symbol("x")[(3,)],
                ),
            ),
        )
    )


def test_builtin_max_empty_no_default():
    x = Symbol("x")
    x.ctx.shapes[x] = (0,)
    x.ctx.types[x] = np.float32
    expr = Constant(max)(x)
    with pytest.raises(ValueError, match=re.escape("max() arg is an empty sequence")):
        _ = SubstituteCalls().visit(expr)


def test_builtin_max_empty_with_default():
    x = Symbol("x")
    x.ctx.shapes[x] = (0,)
    x.ctx.types[x] = np.float32
    expr = Constant(max)(x, default=Constant(-1.0))
    new_expr = SubstituteCalls().visit(expr)
    assert new_expr is Constant(-1.0)


def test_builtin_max_constant():
    a = np.random.randn(5)
    expr = Constant(max)(Constant(a))
    new_expr = SubstituteCalls().visit(expr)
    assert new_expr is not expr
    assert new_expr is Constant(max(a))


def test_builtin_max_constant_with_default():
    a = np.random.randn(5)
    expr = Constant(max)(Constant(a), default=Constant(-1.0))
    new_expr = SubstituteCalls().visit(expr)
    assert new_expr is not expr
    assert new_expr is Constant(max(a, default=-1))

    expr = Constant(max)(Constant(a), default=Constant(0.1))
    new_expr = SubstituteCalls().visit(expr)
    assert new_expr is not expr
    assert new_expr is Constant(max(a, default=0.1))


def test_builtin_max_noshape():
    x = Symbol("x")
    expr = Constant(max)(x)
    new_expr = SubstituteCalls().visit(expr)
    assert new_expr.is_equivalent(expr)


def test_builtin_muliarg_max():
    a = Symbol("a")
    a.ctx.shapes[a] = ()
    a.ctx.types[a] = np.float32
    b = Symbol("b")
    b.ctx.shapes[b] = ()
    b.ctx.types[b] = np.float32
    c = Symbol("c")
    c.ctx.shapes[c] = ()
    c.ctx.types[c] = np.float32
    expr = Constant(max)(a, b, c)
    new_expr = SubstituteCalls().visit(expr)
    assert new_expr is not expr
    assert new_expr.is_equivalent(
        IfThenElse(
            And(GreaterThanOrEqual(a, b), GreaterThanOrEqual(a, c)),
            a,
            IfThenElse(And(GreaterThanOrEqual(b, c)), b, c),
        )
    )


def test_builtin_multiarg_max_constant():
    a = np.random.randn()
    b = np.random.randn()
    c = np.random.randn()
    expr = Constant(max)(Constant(a), Constant(b), Constant(c))
    new_expr = SubstituteCalls().visit(expr)
    assert new_expr is not expr
    assert new_expr is Constant(max(a, b, c))
