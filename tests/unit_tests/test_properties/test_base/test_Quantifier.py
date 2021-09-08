from dnnv.properties.base import *


def test_Quantifier():
    expression = Symbol("x") >= Constant(0)
    quantifier_expression = Quantifier(Symbol("x"), expression)
    assert quantifier_expression.variable is Symbol("x")
    assert quantifier_expression.expression is expression

    quantifier_expression = Quantifier(Symbol("x"), lambda x: x < Constant(0))
    assert quantifier_expression.variable is Symbol("x")
    assert quantifier_expression.expression == (Symbol("x") < 0)


def test_repr():
    expression = Symbol("x") >= Constant(0)
    quantifier_expression = Quantifier(Symbol("x"), expression)
    assert (
        repr(quantifier_expression)
        == "Quantifier(Symbol('x'), GreaterThanOrEqual(Symbol('x'), 0))"
    )

    quantifier_expression = Quantifier(Symbol("x"), lambda x: x < Constant(0))
    assert (
        repr(quantifier_expression)
        == "Quantifier(Symbol('x'), LessThan(Symbol('x'), 0))"
    )


def test_str():
    expression = Symbol("x") >= Constant(0)
    quantifier_expression = Quantifier(Symbol("x"), expression)
    assert str(quantifier_expression) == "Quantifier(x, (x >= 0))"

    quantifier_expression = Quantifier(Symbol("x"), lambda x: x < Constant(0))
    assert str(quantifier_expression) == "Quantifier(x, (x < 0))"
