from dnnv.properties.base import _symbol_from_callable, Symbol


def test_expression():
    symbol = _symbol_from_callable(Symbol("a"))
    assert isinstance(symbol, Symbol)
    assert symbol is Symbol("a")


def test_other():
    symbol = _symbol_from_callable(max)
    assert isinstance(symbol, Symbol)
    assert symbol is Symbol("max")
    assert symbol.is_concrete
    assert symbol.value == max
