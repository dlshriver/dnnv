import ast
import pytest

from dnnv.properties import *
from dnnv.properties.dsl import parse_ast, PropertyParserError


def test_DictComp():
    node = ast.parse(r"{i: 0 for i in range(5)}")
    with pytest.raises(
        PropertyParserError,
        match="DNNP does not currently support dict comprehensions",
    ):
        _ = parse_ast(node)
