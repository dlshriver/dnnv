from dnnv.properties.expressions import *
from dnnv.properties.transformers import PropagateConstants


def test_IfThenElse_symbols():
    transformer = PropagateConstants()

    expr = IfThenElse(Symbol("condition"), Symbol("T"), Symbol("F"))
    new_expr = transformer.visit(expr)
    assert new_expr is not expr
    assert isinstance(new_expr, IfThenElse)
    assert new_expr.expr1 is Symbol("condition")
    assert new_expr.expr2 is Symbol("T")
    assert new_expr.expr3 is Symbol("F")


def test_IfThenElse_consts():
    transformer = PropagateConstants()

    expr = IfThenElse(Constant(True), Constant("t_expr"), Constant("f_expr"))
    new_expr = transformer.visit(expr)
    assert new_expr is not expr
    assert isinstance(new_expr, Constant)
    assert new_expr.value == "t_expr"

    expr = IfThenElse(Constant(False), Constant("t_expr"), Constant("f_expr"))
    new_expr = transformer.visit(expr)
    assert new_expr is not expr
    assert isinstance(new_expr, Constant)
    assert new_expr.value == "f_expr"


def test_IfThenElse_sym_condition_const_expressions():
    transformer = PropagateConstants()

    expr = IfThenElse(Symbol("condition"), Constant("expr"), Constant("expr"))
    new_expr = transformer.visit(expr)
    assert new_expr is not expr
    assert isinstance(new_expr, Constant)
    assert new_expr.value == "expr"

    expr = IfThenElse(Symbol("condition"), Constant(True), Constant(True))
    new_expr = transformer.visit(expr)
    assert new_expr is not expr
    assert isinstance(new_expr, Constant)
    assert new_expr.value == True

    expr = IfThenElse(Symbol("condition"), Constant(True), Constant(False))
    new_expr = transformer.visit(expr)
    assert new_expr is not expr
    assert new_expr is Symbol("condition")

    expr = IfThenElse(Symbol("condition"), Constant(False), Constant(True))
    new_expr = transformer.visit(expr)
    assert new_expr is not expr
    assert isinstance(new_expr, Not)
    assert new_expr.expr is Symbol("condition")


def test_IfThenElse_sym_condition_nonbool_expressions():
    transformer = PropagateConstants()

    expr = IfThenElse(Symbol("condition"), Constant("t_expr"), Constant("f_expr"))
    new_expr = transformer.visit(expr)
    assert new_expr is not expr
    assert isinstance(new_expr, IfThenElse)
    assert new_expr.expr1 is Symbol("condition")
    assert new_expr.expr2 is Constant("t_expr")
    assert new_expr.expr3 is Constant("f_expr")
