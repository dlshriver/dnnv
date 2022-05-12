import pytest

from dnnv.properties import *
from dnnv.properties.parser.dnnp import DNNPParserError, parse_str


def test_Assign_name():
    spec_str = """from dnnv.properties import *
N = Network("N")
N
    """
    phi = parse_str(spec_str)
    with phi.ctx:
        assert phi is Network("N")


def test_Assign_non_name():
    spec_str = """from dnnv.properties import *
networks = [None, None]
networks[0] = Network("N")
networks[0]
    """
    with pytest.raises(
        DNNPParserError,
        match="Assigning to non-identifiers is not currently supported",
    ):
        _ = parse_str(spec_str)


def test_Assign_lambda():
    spec_str = """from dnnv.properties import *
f = lambda x: x + 1
Forall(x, x < f(x))
    """

    phi = parse_str(spec_str)
    with phi.ctx:
        x = Symbol("x")
        assert phi.is_equivalent(Forall(x, x < (x + 1)))
