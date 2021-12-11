from dnnv.properties import *
from dnnv.properties.parser.dnnp import parse_str


def test_Call_non_name():
    spec_str = "f()()"
    phi = parse_str(spec_str)
    with phi.ctx:
        assert phi.is_equivalent(Symbol("f")()())
