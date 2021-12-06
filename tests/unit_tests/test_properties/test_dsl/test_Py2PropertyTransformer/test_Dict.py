import ast
import pytest

from dnnv.properties import *
from dnnv.properties.dsl import parse_ast, PropertyParserError


def test_Dict_empty():
    node = ast.parse(r"{}")
    phi = parse_ast(node)
    assert phi == {}


def test_Dict_non_empty():
    node = ast.parse(r"{0: 1, 1: 2, 'a': 'b', ...: None}")
    phi = parse_ast(node)
    assert phi == {0: 1, 1: 2, "a": "b", ...: None}

    node = ast.parse(r"{0: 0, 1: -1, 2: -2}")
    phi = parse_ast(node)
    assert phi == {0: 0, 1: -1, 2: -2}

    node = ast.parse(r"{'': [], 'a': ['a'], 'ab': ['a', 'b'], 'abc': ['a', 'b', 'c']}")
    phi = parse_ast(node)
    assert phi == {"": [], "a": ["a"], "ab": ["a", "b"], "abc": ["a", "b", "c"]}


def test_Dict_non_primitive():
    node = ast.parse(r"{x: 'x'}")
    with pytest.raises(
        PropertyParserError,
        match="DNNP does not currently support definition of dicts containing non-primitive keys or values",
    ):
        _ = parse_ast(node)

    node = ast.parse(r"{'x': x}")
    with pytest.raises(
        PropertyParserError,
        match="DNNP does not currently support definition of dicts containing non-primitive keys or values",
    ):
        _ = parse_ast(node)
