import ast
import pytest

from dnnv.properties import *
from dnnv.properties.dsl import parse_ast, PropertyParserError


def test_Yield():
    node = ast.parse("yield ...")
    with pytest.raises(
        PropertyParserError, match="DNNP does not support yield expressions"
    ):
        _ = parse_ast(node)
