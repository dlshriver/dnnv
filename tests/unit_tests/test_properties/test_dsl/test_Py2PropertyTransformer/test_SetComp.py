import ast
import pytest

from dnnv.properties import *
from dnnv.properties.dsl import parse_ast, PropertyParserError


def test_SetComp():
    node = ast.parse("{i for i in range(5)}")
    with pytest.raises(
        PropertyParserError,
        match="DNNP does not currently support set comprehensions",
    ):
        _ = parse_ast(node)
