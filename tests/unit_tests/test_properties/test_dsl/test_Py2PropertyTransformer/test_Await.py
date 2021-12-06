import ast
import pytest

from dnnv.properties import *
from dnnv.properties.dsl import parse_ast, PropertyParserError


def test_Await():
    node = ast.parse("await f()")
    with pytest.raises(
        PropertyParserError, match="DNNP does not support await expressions"
    ):
        _ = parse_ast(node)
