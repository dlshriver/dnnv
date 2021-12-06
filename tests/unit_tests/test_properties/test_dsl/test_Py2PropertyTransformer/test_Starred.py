import ast
import pytest

from dnnv.properties import *
from dnnv.properties.dsl import parse_ast, PropertyParserError


def test_Starred():
    node = ast.parse("f(*[1, 2, 3])")
    with pytest.raises(
        PropertyParserError,
        match="DNNP does not currently support starred expressions",
    ):
        _ = parse_ast(node)
