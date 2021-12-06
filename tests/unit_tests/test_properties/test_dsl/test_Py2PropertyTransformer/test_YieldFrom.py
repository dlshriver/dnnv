import ast
import pytest

from dnnv.properties import *
from dnnv.properties.dsl import parse_ast, PropertyParserError


def test_YieldFrom():
    node = ast.parse("yield from ...")
    with pytest.raises(
        PropertyParserError, match="DNNP does not support yield from expressions"
    ):
        _ = parse_ast(node)
