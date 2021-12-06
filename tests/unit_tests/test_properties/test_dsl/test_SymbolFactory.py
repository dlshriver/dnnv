from dnnv.properties import *
from dnnv.properties.dsl import SymbolFactory


def test_get_unknown():
    factory = SymbolFactory(c=0)
    x = factory["x"]

    assert isinstance(x, Symbol)
    assert x is Symbol("x")


def test_get_constant():
    factory = SymbolFactory(c=0)
    zero = factory["c"]

    assert isinstance(zero, Constant)
    assert zero is Constant(0)


def test_get_lambda():
    f_ = lambda x: 2 * x
    factory = SymbolFactory(c=0, f=f_)
    f = factory["f"]

    assert f is f_


def test_get_expression():
    factory = SymbolFactory(c=0, phi=Forall(Symbol("x"), Constant(0) < Symbol("x")))
    phi = factory["phi"]

    assert phi.is_equivalent(Forall(Symbol("x"), Constant(0) < Symbol("x")))
