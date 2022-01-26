from dnnv.properties import *
from dnnv.properties.parser.dnnp import parse_str


def test_Bytes():
    spec_str = 'b"test"'
    phi = parse_str(spec_str)
    assert phi == b"test"


def test_Ellipsis():
    spec_str = "..."
    phi = parse_str(spec_str)
    assert phi == ...


def test_NameConstant():
    spec_str = "None"
    phi = parse_str(spec_str)
    assert phi == None


def test_Num():
    spec_str = "1000"
    phi = parse_str(spec_str)
    assert phi == 1000


def test_Str():
    spec_str = '"test"'
    phi = parse_str(spec_str)
    assert phi == "test"
