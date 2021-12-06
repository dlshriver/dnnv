from dnnv.properties.expressions import *
from dnnv.properties.transformers import PropagateConstants


def test_Call_symbols():
    transformer = PropagateConstants()

    expr = Call(Symbol("x"), (), {})
    new_expr = transformer.visit(expr)
    assert new_expr is not expr
    assert isinstance(new_expr, Call)
    assert new_expr.function is Symbol("x")
    assert new_expr.args == ()
    assert new_expr.kwargs == {}

    expr = Call(Symbol("x"), (Symbol("arg1"),), {})
    new_expr = transformer.visit(expr)
    assert new_expr is not expr
    assert isinstance(new_expr, Call)
    assert new_expr.function is Symbol("x")
    assert new_expr.args == (Symbol("arg1"),)
    assert new_expr.kwargs == {}

    expr = Call(Symbol("x"), (Symbol("arg1"),), {"kw1": Symbol("kwarg1")})
    new_expr = transformer.visit(expr)
    assert new_expr is not expr
    assert isinstance(new_expr, Call)
    assert new_expr.function is Symbol("x")
    assert new_expr.args == (Symbol("arg1"),)
    assert new_expr.kwargs == {"kw1": Symbol("kwarg1")}


def test_Call_consts():
    transformer = PropagateConstants()

    expr = Call(Symbol("f1"), (), {})
    expr.concretize(f1=tuple)
    new_expr = transformer.visit(expr)
    assert new_expr is not expr
    assert isinstance(new_expr, Constant)
    assert new_expr.value == ()

    expr = Call(Symbol("f2"), (Constant(123),), {})
    expr.concretize(f2=str)
    new_expr = transformer.visit(expr)
    assert new_expr is not expr
    assert isinstance(new_expr, Constant)
    assert new_expr.value == "123"

    expr = Call(Symbol("f3"), (Constant("test_name"),), {})
    expr.concretize(f3=lambda name: Symbol(name))
    new_expr = transformer.visit(expr)
    assert new_expr is not expr
    assert isinstance(new_expr, Symbol)
    assert new_expr is Symbol("test_name")
