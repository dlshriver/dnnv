from dnnv.properties import *
from dnnv.properties.parser.dnnp import parse_str


def test_Slice_empty():
    spec_str = "x[:]"
    phi = parse_str(spec_str)
    with phi.ctx:
        x = Symbol("x")
        assert phi.is_equivalent(x[Constant(slice(None))])


def test_Slice_start_end():
    spec_str = "x[b:e]"
    phi = parse_str(spec_str)
    with phi.ctx:
        x = Symbol("x")
        b = Symbol("b")
        e = Symbol("e")
        assert phi.is_equivalent(x[Slice(b, e, Constant(None))])


def test_Slice_start_end_step():
    spec_str = "x[b:e:s]"
    phi = parse_str(spec_str)
    with phi.ctx:
        x = Symbol("x")
        b = Symbol("b")
        e = Symbol("e")
        s = Symbol("s")
        assert phi.is_equivalent(x[Slice(b, e, s)])
