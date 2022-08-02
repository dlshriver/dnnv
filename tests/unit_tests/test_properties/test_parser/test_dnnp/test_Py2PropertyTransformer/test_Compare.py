import pytest

from dnnv.properties import *
from dnnv.properties.parser.dnnp import DNNPParserError, parse_str


def test_Compare_single():
    spec_str = "x < 0"
    phi = parse_str(spec_str)
    with phi.ctx:
        assert phi.is_equivalent(Symbol("x") < Constant(0))

    spec_str = "x <= 0"
    phi = parse_str(spec_str)
    with phi.ctx:
        assert phi.is_equivalent(Symbol("x") <= Constant(0))

    spec_str = "x > 0"
    phi = parse_str(spec_str)
    with phi.ctx:
        assert phi.is_equivalent(Symbol("x") > Constant(0))

    spec_str = "x >= 0"
    phi = parse_str(spec_str)
    with phi.ctx:
        assert phi.is_equivalent(Symbol("x") >= Constant(0))

    spec_str = "x == 0"
    phi = parse_str(spec_str)
    with phi.ctx:
        assert phi.is_equivalent(Symbol("x") == Constant(0))

    spec_str = "x != 0"
    phi = parse_str(spec_str)
    with phi.ctx:
        assert phi.is_equivalent(Symbol("x") != Constant(0))


def test_Compare_multi():
    spec_str = "0 < x < 1"
    phi = parse_str(spec_str)
    with phi.ctx:
        assert phi.is_equivalent(
            And(Constant(0) < Symbol("x"), Symbol("x") < Constant(1))
        )

    spec_str = "0 < x <= 1"
    phi = parse_str(spec_str)
    with phi.ctx:
        assert phi.is_equivalent(
            And(Constant(0) < Symbol("x"), Symbol("x") <= Constant(1))
        )

    spec_str = "a > b > c"
    phi = parse_str(spec_str)
    with phi.ctx:
        assert phi.is_equivalent(
            And(Symbol("a") > Symbol("b"), Symbol("b") > Symbol("c"))
        )

    spec_str = "a >= 0 >= b"
    phi = parse_str(spec_str)
    with phi.ctx:
        assert phi.is_equivalent(
            And(Symbol("a") >= Constant(0), Constant(0) >= Symbol("b"))
        )

    spec_str = "a == b != c"
    phi = parse_str(spec_str)
    with phi.ctx:
        assert phi.is_equivalent(
            And(Symbol("a") == Symbol("b"), Symbol("b") != Symbol("c"))
        )


def test_Compare_unsupported():
    spec_str = "a in b in c"
    with pytest.raises(DNNPParserError, match="Unsupported comparison function:"):
        _ = parse_str(spec_str)
