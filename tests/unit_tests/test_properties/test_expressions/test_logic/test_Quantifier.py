from dnnv.properties.expressions import *


def test_Quantifier():
    expression = Symbol("x") >= Constant(0)
    quantifier_expression = Quantifier(Symbol("x"), expression)
    assert quantifier_expression.variable is Symbol("x")
    assert quantifier_expression.expression is expression


def test_repr():
    expression = Symbol("x") >= Constant(0)
    quantifier_expression = Quantifier(Symbol("x"), expression)
    assert (
        repr(quantifier_expression)
        == "Quantifier(Symbol('x'), GreaterThanOrEqual(Symbol('x'), 0))"
    )


def test_str():
    expression = Symbol("x") >= Constant(0)
    quantifier_expression = Quantifier(Symbol("x"), expression)
    assert str(quantifier_expression) == "Quantifier(x, (x >= 0))"


def test_is_equivalent():
    expr1 = Quantifier(Symbol("a"), Symbol("b"))
    expr2 = Quantifier(Symbol("a"), Symbol("b"))
    expr3 = Quantifier(Symbol("b"), Symbol("a"))
    expr4 = Quantifier(Symbol("a"), Symbol("a"))

    assert expr1.is_equivalent(expr1)
    assert expr1.is_equivalent(expr2)
    assert expr2.is_equivalent(expr1)
    assert not expr1.is_equivalent(expr3)
    assert not expr3.is_equivalent(expr1)
    assert not expr1.is_equivalent(expr4)
    assert not expr4.is_equivalent(expr1)
