import numpy as np
import pytest

from dnnv.properties.expressions import *
from dnnv.properties.transformers import SubstituteCalls


def test_missing():
    class FakeExpression(Expression):
        pass

    with pytest.raises(
        ValueError, match="Unimplemented expression type: FakeExpression"
    ):
        SubstituteCalls().visit(FakeExpression())
    del FakeExpression


def test_Binary_no_func_call():
    transformer = SubstituteCalls()

    expr = Equal(Symbol("a"), Constant(3))
    new_expr = transformer.visit(expr)
    assert new_expr is not expr
    assert isinstance(new_expr, Equal)
    assert new_expr.expr1 is Symbol("a")
    assert new_expr.expr2 is Constant(3)


def test_Binary_unsupported_expr():
    transformer = SubstituteCalls()

    func_call = Call(Symbol("f"), (Symbol("a"),), {})
    func_call.concretize(f=np.argmax)
    expr = Divide(func_call, Constant(3))
    new_expr = transformer.visit(expr)
    assert new_expr is not expr
    assert isinstance(new_expr, Divide)
    assert str(new_expr.expr1) == "f(a)"
    assert new_expr.expr2 is Constant(3)

    expr = Subtract(Constant(0), func_call)
    new_expr = transformer.visit(expr)
    assert new_expr is not expr
    assert isinstance(new_expr, Subtract)
    assert new_expr.expr1 is Constant(0)
    assert str(new_expr.expr2) == "f(a)"


def test_Binary_not_implemented():
    transformer = SubstituteCalls()

    func_call = Call(Symbol("f"), (Symbol("a"),), {})
    func_call.concretize(f=np.argmax)
    expr = Equal(func_call, Constant(3))
    new_expr = transformer.visit(expr)
    assert new_expr is not expr
    assert isinstance(new_expr, Equal)
    assert str(new_expr.expr1) == "f(a)"
    assert new_expr.expr2 is Constant(3)

    expr = NotEqual(Constant(0), func_call)
    new_expr = transformer.visit(expr)
    assert new_expr is not expr
    assert isinstance(new_expr, NotEqual)
    assert new_expr.expr1 is Constant(0)
    assert str(new_expr.expr2) == "f(a)"


def test_Binary():
    transformer = SubstituteCalls()

    func_call = Call(Symbol("f"), (Network("N")(Symbol("a")),), {})
    fake_network = lambda x: x
    fake_network.output_shape = ((1, 3),)
    func_call.concretize(f=np.argmax, N=fake_network)
    expr = Equal(func_call, Constant(5))
    new_expr = transformer.visit(expr)
    assert new_expr is not expr
    assert isinstance(new_expr, Constant)
    assert new_expr is Constant(False)

    expr = NotEqual(Constant(0), func_call)
    new_expr = transformer.visit(expr)
    assert new_expr is not expr
    assert (
        str(new_expr)
        == "((N(a)[(0, 0)] < N(a)[(0, 1)]) | (N(a)[(0, 0)] < N(a)[(0, 2)]))"
    )


def test_Call():
    transformer = SubstituteCalls()

    func_call = Call(Constant(np.argmax), (Network("N")(Symbol("a")),), {})
    fake_network = lambda x: x
    fake_network.output_shape = ((1, 3),)
    func_call.concretize(N=fake_network)
    new_expr = transformer.visit(func_call)
    assert new_expr is not func_call
    assert isinstance(new_expr, IfThenElse)
    assert (
        str(new_expr)
        == "IfThenElse(((N(a)[(0, 0)] >= N(a)[(0, 1)]) & (N(a)[(0, 0)] >= N(a)[(0, 2)])), 0, IfThenElse(((N(a)[(0, 1)] >= N(a)[(0, 2)])), 1, 2))"
    )

    func_call = Call(Symbol("f"), (Network("N")(Symbol("a")),), {})
    fake_network = lambda x: x
    fake_network.output_shape = ((1, 3),)
    func_call.concretize(f=np.argmax, N=fake_network)
    new_expr = transformer.visit(func_call)
    assert new_expr is not func_call
    assert isinstance(new_expr, IfThenElse)
    assert (
        str(new_expr)
        == "IfThenElse(((N(a)[(0, 0)] >= N(a)[(0, 1)]) & (N(a)[(0, 0)] >= N(a)[(0, 2)])), 0, IfThenElse(((N(a)[(0, 1)] >= N(a)[(0, 2)])), 1, 2))"
    )
