from dnnv.properties import *
from dnnv.properties.parser.dnnp import parse_str


def test_IfExp():
    spec_str = "T if x else F"
    phi = parse_str(spec_str)
    with phi.ctx:
        assert phi.is_equivalent(IfThenElse(Symbol("x"), Symbol("T"), Symbol("F")))
