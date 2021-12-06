import ast
import pytest

from dnnv.properties import *
from dnnv.properties.dsl import parse_ast, PropertyParserError


def test_Set_non_empty():
    node = ast.parse("{0, 1, 2, 3, 4}")
    phi = parse_ast(node)
    assert phi == {0, 1, 2, 3, 4}


def test_Set_non_primitive():
    node = ast.parse("{'a', x}")
    with pytest.raises(
        PropertyParserError,
        match="DNNP does not currently support definition of sets containing non-primitive types",
    ):
        _ = parse_ast(node)
