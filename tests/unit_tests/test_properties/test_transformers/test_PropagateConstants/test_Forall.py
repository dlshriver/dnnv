from dnnv.properties.expressions import *
from dnnv.properties.transformers import PropagateConstants


def test_Forall_symbols():
    transformer = PropagateConstants()

    expr = Forall(Symbol("x"), Symbol("x"))
    new_expr = transformer.visit(expr)
    assert new_expr is not expr
    assert isinstance(new_expr, Forall)
    assert new_expr.variable is Symbol("x")
    assert new_expr.expression is Symbol("x")


def test_Forall_consts():
    transformer = PropagateConstants()

    expr = Forall(Symbol("x"), Constant(False))
    new_expr = transformer.visit(expr)
    assert new_expr is not expr
    assert isinstance(new_expr, Constant)
    assert new_expr.value == False

    expr = Forall(Symbol("x"), Constant(True))
    new_expr = transformer.visit(expr)
    assert new_expr is not expr
    assert isinstance(new_expr, Constant)
    assert new_expr.value == True


def test_Forall_unused_variable():
    transformer = PropagateConstants()

    expr = Forall(Symbol("x"), Symbol("y"))
    new_expr = transformer.visit(expr)
    assert new_expr is not expr
    assert isinstance(new_expr, Symbol)
    assert new_expr is Symbol("y")
