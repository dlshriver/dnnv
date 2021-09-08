from dnnv.properties.base import *
from dnnv.properties.base import find_symbols


def test_0():
    phi = Symbol("a")
    symbols = find_symbols(phi)
    assert len(symbols) == 1
    assert Symbol("a") in symbols


def test_1():
    phi = And(Symbol("a"), Symbol("b"), Constant(True))
    symbols = find_symbols(phi)
    assert len(symbols) == 2
    assert Symbol("a") in symbols
    assert Symbol("b") in symbols


def test_2():
    phi = LessThan(Symbol("a"), Constant(0))
    symbols = find_symbols(phi)
    assert len(symbols) == 1
    assert Symbol("a") in symbols


def test_3():
    phi = Constant(
        {
            0: Symbol("a"),
            1: Symbol("b"),
            2: "c",
            3: Constant("d"),
            Constant(4): "e",
            Symbol("key:5"): "f",
        }
    )
    symbols = find_symbols(phi)
    assert len(symbols) == 3
    assert Symbol("a") in symbols
    assert Symbol("b") in symbols
    assert Symbol("key:5") in symbols
