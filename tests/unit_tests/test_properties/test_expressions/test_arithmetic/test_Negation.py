from dnnv.properties.expressions import *


def test_Negation_symbols():
    a = Symbol("a")

    c_1 = -a
    c_2 = Negation(a)

    assert c_1.is_equivalent(c_2)
    assert isinstance(c_1, ArithmeticExpression)


def test_Negation_constants():
    a = Constant(1)

    c_1 = -a
    c_2 = Negation(a)
    assert c_1.is_equivalent(c_2)
    assert isinstance(c_1, ArithmeticExpression)
    assert c_1.is_concrete
    assert c_1.value == -1
