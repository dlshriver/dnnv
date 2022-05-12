import pytest

from dnnv.properties import *
from dnnv.properties.parser.dnnp import DNNPParserError, parse_str


def test_Dict_empty():
    spec_str = r"{}"
    phi = parse_str(spec_str)
    assert phi == {}


def test_Dict_non_empty():
    spec_str = r"{0: 1, 1: 2, 'a': 'b', ...: None}"
    phi = parse_str(spec_str)
    assert phi == {0: 1, 1: 2, "a": "b", ...: None}

    spec_str = r"{0: 0, 1: -1, 2: -2}"
    phi = parse_str(spec_str)
    assert phi == {0: 0, 1: -1, 2: -2}

    spec_str = r"{'': [], 'a': ['a'], 'ab': ['a', 'b'], 'abc': ['a', 'b', 'c']}"
    phi = parse_str(spec_str)
    assert phi == {"": [], "a": ["a"], "ab": ["a", "b"], "abc": ["a", "b", "c"]}


def test_Dict_non_primitive():
    spec_str = r"{x: 'x'}"
    with pytest.raises(
        DNNPParserError,
        match="DNNP does not currently support definition of dicts containing non-primitive keys or values",
    ):
        _ = parse_str(spec_str)

    spec_str = r"{'x': x}"
    with pytest.raises(
        DNNPParserError,
        match="DNNP does not currently support definition of dicts containing non-primitive keys or values",
    ):
        _ = parse_str(spec_str)
